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
# * reply去重 keep='first' 根据query  0.78709343009  epoch=4 
# * reply去重 keep=False  根据id     0.78437459578  epoch=4
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 6 --max_length 100 --cuda 3 --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE-TAPT --attack_type fgm --scheduler --cross_val_fold 7

# python train.py --model_name bert_spc --seed 8888 --bert_lr 2e-5 --num_epoch 6 --max_length 100 --cuda 3 --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/chinese_bert_wwm_ext_pytorch --attack_type fgm --scheduler --cross_val_fold 7

# python train.py --model_name bert_spc --seed 8888 --bert_lr 2e-5 --num_epoch 6 --max_length 100 --cuda 3 --notsavemodel --log_step 20 --pretrained_bert_name hfl/chinese-bert-wwm --attack_type fgm --scheduler --cross_val_fold 7

python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3 --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE-TAPT --attack_type fgm --scheduler --cross_val_fold 5 --add_pseudo_data --pos_num 5000 --pseudo_path ./results/bert_spc_ERNIE-TAPT_FGM_7-fold_20201019/kfold/voted.tsv

# python train.py --model_name bert_rnn --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 2  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE-TAPT --attack_type fgm --scheduler --cross_val_fold 7 --diff_lr --layers_lr 0.001

# * roberta-large kfold=5
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/chinese_roberta_wwm_large_ext_pytorch --scheduler --cross_val_fold 5 --bert_dim 1024 --train_batch_size 10

# * clue/roberta_chinese_pair_large
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 4 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name clue/roberta_chinese_pair_large --scheduler --cross_val_fold 5 --bert_dim 1024 --train_batch_size 10
# * 0.77749967066
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 4 --max_length 100 --cuda 2  --notsavemodel --log_step 20 --pretrained_bert_name clue/roberta_chinese_pair_large --scheduler --cross_val_fold 5 --bert_dim 1024 --train_batch_size 10 --attack_type fgm

# * ERNIE-TAPT fgm + kfold=7 diff_lr focalloss 0.78560042508
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE-TAPT --attack_type fgm --scheduler --cross_val_fold 7 --diff_lr --criterion focalloss

# * ERNIE-TAPT fgm + kfold=7 0.78492475696
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 2  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE-TAPT --attack_type fgm --scheduler --cross_val_fold 7 --criterion ghmc

# * ERNIE-TAPT fgm + kfold=7 batch=32 0.78496868476
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE-TAPT --attack_type fgm --scheduler --cross_val_fold 7 --train_batch_size 32

# * ERNIE-TAPT fgm + kfold=7 datareverse  0.78575129534
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 1  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE-TAPT --attack_type fgm --scheduler --cross_val_fold 7 --datareverse

# * ERNIE-TAPT fgm + kfold=5 order_predict  0.77677520596
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 1  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE-TAPT --attack_type fgm --scheduler --cross_val_fold 5 --order_predict

# * ERNIE-TAPT fgm + kfold=5 bert_cap diff_lr  0.78326013950
# python train.py --model_name bert_cap --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE-TAPT --attack_type fgm --scheduler --train_batch_size 8 --diff_lr

# * bert_wwm_ext
# python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/chinese_bert_wwm_ext_pytorch --attack_type fgm --scheduler --cross_val_fold 7 --train_batch_size 32

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