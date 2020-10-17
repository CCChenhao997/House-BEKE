#!/bin/bash

# * bert-base-chinese
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 5

# * ERNIE 效果明显比 bert-base-chinese 要好  0.77308091 BCEloss
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE --datareverse

python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 5 --max_length 100 --cuda 1  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/chinese_roberta_wwm_ext_pytorch

python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 4 --max_length 100 --cuda 1  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/chinese_roberta_wwm_large_ext_pytorch --bert_dim 1024

# * ghmc 0.77179287015 
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE --criterion ghmc

# * flooding需要调参，0.2很差
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 1 --max_length 100 --cuda 3 --notsavemodel --log_step 100 --flooding 0.2

# * focalloss 不如 BCE
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 1 --max_length 100 --cuda 3  --notsavemodel --log_step 100 --criterion focalloss