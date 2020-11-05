import torch
import torch.nn as nn


class Bert_Spc_PET(nn.Module):
    def __init__(self, bert, opt):
        super(Bert_Spc_PET, self).__init__()
        self.opt = opt
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        layers = [nn.Linear(
            opt.bert_dim, opt.bert_dim // 2), nn.ReLU(), nn.Linear(opt.bert_dim // 2, opt.label_dim)]
        self.dense = nn.Sequential(*layers)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask = inputs[0], inputs[1], inputs[2]
        _, pooled_output, all_hidden_states = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.dense(pooled_output)
        
        last_layer_hidden_states = all_hidden_states[-1]
        pet_hidden = last_layer_hidden_states[:, 1, :]
        pet_hidden = self.dropout(pet_hidden)
        logits = self.dense(pet_hidden)

        return logits