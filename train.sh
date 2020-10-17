#!/bin/bash

# * bert-base-chinese
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 5

# * ERNIE 效果明显比 bert-base-chinese 要好
python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE

# * flooding需要调参，0.2很差
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 1 --max_length 100 --cuda 3 --notsavemodel --log_step 100 --flooding 0.2

# * focalloss 不如 BCE
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 1 --max_length 100 --cuda 3  --notsavemodel --log_step 100 --criterion focalloss