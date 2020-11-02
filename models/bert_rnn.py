import torch
import torch.nn as nn
from dynamic_rnn import DynamicLSTM


class Bert_RNN(nn.Module):
    def __init__(self, bert, opt):
        super(Bert_RNN, self).__init__()
        self.opt = opt
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        layers = [nn.Linear(
            opt.bert_dim, opt.bert_dim // 2), nn.ReLU(), nn.Linear(opt.bert_dim // 2, opt.label_dim)]
        self.dense = nn.Sequential(*layers)
        self.rnn = DynamicLSTM(opt.bert_dim, opt.bert_dim // 4, num_layers=1, batch_first=True, bidirectional=True, rnn_type='LSTM')
        # self.rnn = nn.LSTM(opt.bert_dim, opt.bert_dim // 4, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask = inputs[0], inputs[1], inputs[2]
        sentence_output, pooled_output, all_hidden_states = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        sentence_output = self.dropout(sentence_output)

        text_len = torch.sum(text_bert_indices != 0, dim=-1)
        rnn_out, _ = self.rnn(sentence_output, text_len)
        birnn_output = torch.cat((rnn_out[:, 0], rnn_out[:, -1]), -1)

        logits = self.dense(birnn_output)
        logits = torch.sigmoid(logits)
        
        return logits