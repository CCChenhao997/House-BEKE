import torch
import torch.nn as nn
import copy

from torch.nn.modules import rnn


class Dual_Bert(nn.Module):
    def __init__(self, bert, opt):
        super(Dual_Bert, self).__init__()
        self.opt = opt
        self.bert_query = bert
        self.bert_reply = copy.deepcopy(bert)
        self.dropout_query = nn.Dropout(opt.dropout_query)
        self.dropout_reply = nn.Dropout(opt.dropout_reply)
        layers = [nn.Linear(
            opt.bert_dim*2, opt.bert_dim), nn.ReLU(), nn.Linear(opt.bert_dim, opt.label_dim)]
        self.dense = nn.Sequential(*layers)
        self.order_dense = nn.Linear(opt.bert_dim, opt.order_dim)
        self.rnn = nn.GRU(opt.bert_dim, opt.bert_dim//2, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, inputs):
        query_indices, reply_indices, attention_mask_query, attention_mask_reply = inputs[0], inputs[1], inputs[2], inputs[3]
        _, pooled_query, _ = self.bert_query(query_indices, attention_mask=attention_mask_query)
        _, pooled_reply, _ = self.bert_reply(reply_indices, attention_mask=attention_mask_reply)

        pooled_query = self.dropout_query(pooled_query).unsqueeze(1)    # (16, 1, 768)
        pooled_reply = self.dropout_reply(pooled_reply).unsqueeze(1)

        pair = torch.cat((pooled_query, pooled_reply), dim=1)       # (16, 2, 768)

        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(pair)                                 # (16, 2, 768)
        rnn_cat = torch.cat((rnn_out[:, 0, :], rnn_out[:, 1, :]), dim=-1)   # (16, 768*2)
        logits = self.dense(rnn_cat)
        
        return logits