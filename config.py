import os
import sys
import torch
import torch.nn as nn
import argparse
import logging
from models.bert_spc import Bert_Spc
from models.bert_cap import Bert_Cap
from models.bert_rnn import Bert_RNN
from models.dual_bert import Dual_Bert


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

dataset_files = {
        'beke': {
            'train_query': './data/train/train.query.tsv',
            'train_reply': './data/train/train.reply.tsv',
            'test_query': './data/test/test.query.tsv',
            'test_reply': './data/test/test.reply.tsv',
        },
    }

model_classes = {
        'bert_spc': Bert_Spc,
        'bert_cap': Bert_Cap,
        'bert_rnn': Bert_RNN,
        'dual_bert': Dual_Bert,
    }

input_colses = {
        'bert_spc': ['dialogue_pair_indices', 'bert_segments_ids', 'attention_mask'],
        'bert_cap': ['dialogue_pair_indices', 'bert_segments_ids', 'attention_mask'],
        'bert_rnn': ['dialogue_pair_indices', 'bert_segments_ids', 'attention_mask'],
        'dual_bert': ['query_indices', 'reply_indices', 'attention_mask_query', 'attention_mask_reply']
    }
    
initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,    # default lr=0.01
        'adam': torch.optim.Adam,          # default lr=0.001
        'adamax': torch.optim.Adamax,      # default lr=0.002
        'asgd': torch.optim.ASGD,          # default lr=0.01
        'rmsprop': torch.optim.RMSprop,    # default lr=0.01
        'sgd': torch.optim.SGD,
    }


# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='bert_spc', type=str, help=', '.join(model_classes.keys()))
parser.add_argument('--dataset', default='beke', type=str)
parser.add_argument('--bert_lr', default=2e-5, type=float)    # 1e-3
parser.add_argument('--layers_lr', default=0.001, type=float)
parser.add_argument('--diff_lr', default=False, action='store_true')
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--dropout_query', default=0.3, type=float)
parser.add_argument('--dropout_reply', default=0.3, type=float)
parser.add_argument('--l2reg', default=1e-5, type=float)    # 1e-5
parser.add_argument('--num_epoch', default=5, type=int)
parser.add_argument('--train_batch_size', default=16, type=int)
parser.add_argument('--eval_batch_size', default=32, type=int)
parser.add_argument('--log_step', default=5, type=int)
parser.add_argument('--hidden_dim', default=300, type=int)
parser.add_argument('--label_dim', default=1, type=int)
parser.add_argument('--max_length', default=100, type=int)
parser.add_argument('--dia_maxlength', default=32, type=int)
parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
parser.add_argument('--repeats', default=1, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--bert_dim', default=768, type=int)
parser.add_argument('--pretrained_bert_name', default='bert-base-chinese', type=str)
parser.add_argument("--weight_decay", default=0.00, type=float, help="Weight deay if we apply some.") # 0.01
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument('--cross_val_fold', default=5, type=int, help='k-fold cross validation')
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='clip gradients at this value')
parser.add_argument('--cuda', default='0', type=str)
parser.add_argument('--datatype', default=None, type=str, choices=['transdata', 'diadata', 'raw'])
parser.add_argument('--attention_hops', default=5, type=int)
parser.add_argument('--attack_type', default=None, type=str, help='fgm, pgd')
parser.add_argument('--criterion', default=None, type=str, help='loss choice', choices=['focalloss', 'ghmc', 'diceloss'])
parser.add_argument('--alpha', default=0.25, type=float)
parser.add_argument('--gamma', default=2, type=int)
parser.add_argument('--smooth', default=0.1, type=float)
parser.add_argument('--threshold', default=0.50, type=float)
parser.add_argument('--flooding', default=0, type=float)
parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')
parser.add_argument('--date', default='date', type=str)
parser.add_argument('--add_pseudo_data', default=False, action='store_true')
parser.add_argument('--pseudo_path', default=None, type=str)
parser.add_argument('--pos_num', default=5000, type=int)
parser.add_argument('--neg_num', default=0, type=int)
parser.add_argument('--rnntype', default='LSTM', type=str, choices=['LSTM', 'GRU', 'RNN'])
parser.add_argument('--scheduler', default=False, action='store_true')
parser.add_argument('--notsavemodel', default=False, action='store_true')
parser.add_argument('--datareverse', default=False, action='store_true')
parser.add_argument('--order_predict', default=False, action='store_true')
parser.add_argument('--order_dim', default=1, type=int)
parser.add_argument('--start', default=20, type=int)
parser.add_argument('--end', default=60, type=int)
opt = parser.parse_args()
opt.model_class = model_classes[opt.model_name]
opt.inputs_cols = input_colses[opt.model_name]
os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)