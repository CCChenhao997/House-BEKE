#!/bin/bash

# * bert-base-chinese
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 5

# * ERNIE 效果明显比 bert-base-chinese 要好  0.77308091 BCEloss
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE

# * fgm 0.77975341
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 4 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE --attack_type fgm --scheduler

# * fgm + kfold=7  0.78033390298
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE --attack_type fgm --scheduler --cross_val_fold 7

# * ERNIE-TAPT fgm + kfold=7  0.78855493
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE-TAPT --attack_type fgm --scheduler --cross_val_fold 7

# * ERNIE-TAPT fgm + kfold=7 datareverse  0.78575129534
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 1  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE-TAPT --attack_type fgm --scheduler --cross_val_fold 7 --datareverse

# * ERNIE-TAPT fgm + kfold=5 datareverse & order_predict 训练巨慢，什么原因？
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 2 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE-TAPT --attack_type fgm --scheduler --cross_val_fold 5 --datareverse --order_predict

# * ERNIE-TAPT fgm + kfold=5 bert_cap diff_lr
python train.py --model_name bert_cap --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE-TAPT --attack_type fgm --scheduler --train_batch_size 8 --diff_lr

# * bert_wwm_ext
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 2  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/chinese_bert_wwm_ext_pytorch --scheduler --cross_val_fold 7

# * pgd 0.77839228296 训练时间要比fgm长
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 4 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE --attack_type pgd --scheduler

# * diceloss
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE --scheduler --criterion diceloss

# * datareverse 0.77587103484 
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE --datareverse

# * ghmc 0.77179287015 
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE --criterion ghmc

# * flooding需要调参，0.2很差
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 1 --max_length 100 --cuda 3 --notsavemodel --log_step 100 --flooding 0.2

# * focalloss 不如 BCE
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 1 --max_length 100 --cuda 3  --notsavemodel --log_step 100 --criterion focalloss