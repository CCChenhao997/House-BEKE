## 房产行业聊天问答匹配

### A榜 48/2985

#### Tricks

- [x] F1 search
- [ ] Each fold model uses the same threshold
- [x] Focalloss
- [x] GHM-C loss
- [x] DiceLoss
- [x] FGM attack
- [x] PGD attack
- [x] K-fold voting (5-7-9 fold)
- [ ] Fusion of multiple models
- [ ] Pseudo label [[reference]](https://github.com/zzy99/epidemic-sentence-pair)
- [x] Optimizers
- [ ] Multiple loss: accelerate convergence in the early stage and improve accuracy in the later stage.
- [ ] Reverse query-reply

#### Methods

- [x] Bert
- [x] Double bert + LSTM (Double bert model query and reply respectively, then a BiLSTM model the relation between them)
- [ ] LCF [[link]](https://www.mdpi.com/2076-3417/9/16/3389)
- [ ] Semantic Role Labeling
- [x] Auxiliary task: Identify query-reply or reply-query
- [x] Bert+GCN

#### Pre-training language models

- [x] ERNIE 1.0
- [x] RoBERTa
- [x] BERT-wwm-ext
- [x] Pre-training with test dataset
- [ ] Pre-training with QA data in the real estate field
- [ ] Pre-training with QA data in other domain [[Link1]](https://spaces.ac.cn/archives/4338/comment-page-2#comments) [[Link2]](https://github.com/chatopera/insuranceqa-corpus-zh)

#### Data processing

- [x] Truncation
- [x] Clean
- [x] KFold
- [x] StratifiedKFold
- [x] GroupKFold
- [x] Exchange query-reply pair order
- [x] Delete duplicate queries and emoji
- [ ] Cluster analysis for queries & Multi-task learning
- [ ] Back translation
- [ ] Split reply that exceeds the maximum length
- [ ] LaserTagger [[link]](https://github.com/Mleader2/text_scalpel)
- [ ] Random Mask

#### Papers

- [ ] SemBERT [[link]](https://arxiv.org/abs/1909.02209)
- [ ] Explicit Contextual Semantics for Text Comprehension [[link]](https://arxiv.org/abs/1809.02794)

#### Analysis

- [x] Prediction results of dev set
- [ ] Prediction results of test set

-----------------------------

### Submission history

1. 0.77308091 | bert_spc | ERNIE | Search_f1 | StratifiedKFold 5-fold voting | **BCE loss**

   `python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE`

2. 0.77179287015 | bert_spc | ERNIE | Search_f1 | StratifiedKFold 5-fold voting | **GHMC loss**

   `python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE --criterion ghmc`

3. 0.77587103484 | bert_spc | ERNIE | Search_f1 | StratifiedKFold 5-fold voting | BCE loss | **datareverse**

   `python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE --datareverse`

4. **0.77975341** | bert_spc | ERNIE | Search_f1 | StratifiedKFold 5-fold voting | BCE loss | **FGM**

   `python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 4 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE --attack_type fgm --scheduler`

5. 0.77839228296 | bert_spc | ERNIE | Search_f1 | StratifiedKFold 5-fold voting | BCE loss | **PGD**

   `python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 4 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE --attack_type pgd --scheduler`

6. **0.78033390298** | bert_spc | ERNIE | Search_f1 | StratifiedKFold **7-fold** voting | BCE loss | **FGM**

   `python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE --attack_type fgm --scheduler --cross_val_fold 7`

7. **0.78855493** | bert_spc | **ERNIE-TAPT** | Search_f1 | StratifiedKFold **7-fold** voting | BCE loss | **FGM**

   `python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE-TAPT --attack_type fgm --scheduler --cross_val_fold 7`

8. 0.78575129534 | bert_spc | ERNIE-TAPT | Search_f1 | StratifiedKFold **7-fold** voting | BCE loss | FGM | **datareverse**

   `python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 1  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE-TAPT --attack_type fgm --scheduler --cross_val_fold 7 --datareverse`

9. 0.77677520596 | bert_spc | ERNIE-TAPT | Search_f1 | BCE loss | FGM | **order_predict**

   `python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 1  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE-TAPT --attack_type fgm --scheduler --cross_val_fold 5 --order_predict`

10. 0.78326013950 | **bert_cap** | ERNIE-TAPT | Search_f1 | BCE loss | FGM | batchsize=8 | diff_lr

    `python train.py --model_name bert_cap --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE-TAPT --attack_type fgm --scheduler --train_batch_size 8 --diff_lr`

11. 0.78496868476 | bert_spc | ERNIE-TAPT | Search_f1 | StratifiedKFold 7-fold voting | BCE loss | FGM | **batchsize=32**

    `python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE-TAPT --attack_type fgm --scheduler --cross_val_fold 7 --train_batch_size 32`
    
12. **0.79014267185** | bert_spc | ERNIE-ALL-TAPT | Search_f1 | GroupKFold 7-fold voting | BCE loss | FGM 

    `python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 4 --max_length 100 --cuda 3 --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE-ALL-TAPT --attack_type fgm --scheduler --cross_val_fold 7 --cv_type GroupKFold`

13. **0.79081172** | bert_spc | ERNIE-ALL-TAPT | Search_f1 | GroupKFold 7-fold voting | GHMC loss | FGM 

    `python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 4 --max_length 100 --cuda 2 --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE-ALL-TAPT --attack_type fgm --scheduler --cross_val_fold 7 --cv_type GroupKFold --criterion ghmc`

----------------------

### Licence

[MIT](./LICENSE)

