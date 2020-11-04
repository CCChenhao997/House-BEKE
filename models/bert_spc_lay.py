import torch
import torch.nn as nn


class Bert_Spc_Lay(nn.Module):
    def __init__(self, bert, opt):
        super(Bert_Spc_Lay, self).__init__()
        self.opt = opt
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        layers = [nn.Linear(
            opt.bert_dim*3, opt.bert_dim), nn.ReLU(), nn.Linear(opt.bert_dim, opt.label_dim)]
        self.dense = nn.Sequential(*layers)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask = inputs[0], inputs[1], inputs[2]
        _, pooled_output, all_hidden_states = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)

        last_three_hidden_states = all_hidden_states[-3:]
        concated_layers = torch.cat(last_three_hidden_states, dim=-1)
        concated_layers_cls = concated_layers[:, 0, :]
        concated_layers_cls = self.dropout(concated_layers_cls)

        logits = self.dense(concated_layers_cls)
        logits = self.dense(pooled_output)
        # logits = torch.sigmoid(logits)
        
        return logits