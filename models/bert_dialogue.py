import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bert_Dialogue(nn.Module):
    def __init__(self, bert, opt):
        super(Bert_Dialogue, self).__init__()
        self.opt = opt
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        layers = [nn.Linear(
            opt.bert_dim, opt.bert_dim // 2), nn.ReLU(), nn.Linear(opt.bert_dim // 2, opt.label_dim)]
        self.dense = nn.Sequential(*layers)

        # self.W = nn.ModuleList()
        # for layer in range(self.opt.GCN_layers):
        #     # input_dim = self.in_dim if layer == 0 else self.mem_dim
        #     # self.W.append(nn.Linear(input_dim, self.mem_dim))
        #     self.W.append(nn.Linear(opt.bert_dim, opt.bert_dim))
        
        self.heads = opt.heads
        # self.head_dim = self.mem_dim // self.opt.GCN_layers
        # self.head_dim = opt.bert_dim // self.opt.GCN_layers
        # self.attn = MultiHeadAttention(self.heads, self.mem_dim*2)
        self.attn = MultiHeadAttention(self.heads, opt.bert_dim)
        self.weight_list = nn.ModuleList()
        for i in range(self.heads):
            for j in range(self.opt.GCN_layers):
                self.weight_list.append(nn.Linear(opt.bert_dim, opt.bert_dim))

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask = inputs[0], inputs[1], inputs[2]
        _, pooled_output, all_hidden_states = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        pooled_output = self.dropout(pooled_output)
        pooled_output_replies = pooled_output.unsqueeze(0)   # (1, n, 768)

        # logits = self.dense(pooled_output)
        attn_tensor = self.attn(pooled_output_replies, pooled_output_replies)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        multi_head_list = []
        adj = None
        for i in range(self.heads):
            adj = attn_adj_list[i]
            # ********* adj_ag对角线清零后加1 **********
            for j in range(adj.size(0)):
                adj[j] -= torch.diag(torch.diag(adj[j]))
                adj[j] += torch.eye(adj[j].size(0)).cuda()
            # adj = mask_ * adj
            # ****************************************
            denom_ag = adj.sum(2).unsqueeze(2) + 1
            outputs = pooled_output_replies
            # output_list = []
            for l in range(self.opt.GCN_layers):
                index = i * self.opt.GCN_layers + l
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW / denom_ag
                gAxW = F.relu(AxW)
                outputs = self.gcn_drop(gAxW) if l < self.opt.GCN_layers - 1 else gAxW
                # output_list.append(self.gcn_drop(gAxW))

            multi_head_list.append(outputs)

        gcn_output = torch.cat(multi_head_list, dim=2) if len(multi_head_list) > 1 else multi_head_list[0]
        gcn_output = gcn_output.squeeze(0)
        logits = self.dense(gcn_output)
        
        penal = None
        if self.opt.regular:
        # * 正交正则
            adj_T = adj.transpose(1, 2)
            identity = torch.eye(adj.size(1)).cuda()
            identity = identity.unsqueeze(0).expand(adj.size(0), adj.size(1), adj.size(1))
            ortho = adj@adj_T

            for i in range(ortho.size(0)):
                ortho[i] -= torch.diag(torch.diag(ortho[i]))
                ortho[i] += torch.eye(ortho[i].size(0)).cuda()

            penal = (torch.norm(ortho - identity) / adj.size(0)).cuda()
            penal = self.opt.penal_weight * penal
        
        return logits, penal


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask[:, :, :query.size(1)]
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)    # 16
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn