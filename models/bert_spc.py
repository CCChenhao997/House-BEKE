import torch
import torch.nn as nn


class Bert_Spc(nn.Module):
    def __init__(self, bert, opt):
        super(Bert_Spc, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        # self.dense = Linear(opt.bert_dim, opt.label_dim)
        layers = [nn.Linear(
            opt.bert_dim, 256), nn.ReLU(), nn.Linear(256, opt.label_dim)]
        self.dense = nn.Sequential(*layers)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask = inputs[0], inputs[1], inputs[2]
        _, pooled_output, all_hidden_states = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits