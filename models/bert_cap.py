import torch
import torch.nn as nn
import torch.nn.functional as F
from capsule import capsule_fusion


class Bert_Cap(nn.Module):
    def __init__(self, bert, opt):
        super(Bert_Cap, self).__init__()
        self.opt = opt
        self.capsule_hidden_size = 128

        self.layer_fusion = capsule_fusion(opt=opt, D=opt.bert_dim,
                                           n_in=3, n_out=opt.bert_dim,
                                           in_dim=128, out_dim=1,
                                           depth_encoding=True)

        self.hidden_fusion = capsule_fusion(opt=opt, D=opt.bert_dim,
                                            n_in=1, n_out=2*self.capsule_hidden_size,
                                            in_dim=128, out_dim=1,
                                            depth_encoding=False)

        self.linear1 = nn.Linear(2*self.capsule_hidden_size, 300)

        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        layers = [nn.Linear(
            opt.bert_dim+300, opt.bert_dim // 2), nn.ReLU(), nn.Linear(opt.bert_dim // 2, opt.label_dim)]
        self.dense = nn.Sequential(*layers)
        # self.linear = Linear(opt.bert_dim + 300, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask = inputs[0], inputs[1], inputs[2]
        sentence_output, pooled_output, all_hidden_states = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        pooled_output = self.dropout(pooled_output)

        N, S = text_bert_indices.size()

        last_six_hidden_states = all_hidden_states[-3:]
        concated_layers = torch.cat(last_six_hidden_states, dim=-1)
        concated_layers = concated_layers.view(N*S, 3, self.opt.bert_dim)

        _, fused_layers = self.layer_fusion(concated_layers)

        fused_layers = fused_layers.view(N*S, self.opt.bert_dim)
        fused_layers = fused_layers.view(N, S, self.opt.bert_dim)

        fused_layers = self.dropout(fused_layers)

        _, fused_hidden = self.hidden_fusion(fused_layers)

        fused_hidden = fused_hidden.view(N, 2*self.capsule_hidden_size)

        intermediate = F.gelu(self.linear1(fused_hidden))
        final_output = torch.cat((pooled_output, intermediate), dim=-1)
        logits = self.dense(final_output)
        # logits = torch.sigmoid(logits)

        return logits