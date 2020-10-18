## 房产行业聊天问答匹配

### TODO

#### Tricks

- [x] F1 search
- [ ] Early stop
- [x] Focalloss
- [x] GHM-C loss
- [x] DiceLoss
- [x] FGM attack
- [x] PGD attack
- [x] K-fold voting (5-7-9 fold)
- [ ] Fusion of multiple models
- [ ] Pseudo label [[reference]](https://github.com/zzy99/epidemic-sentence-pair)

#### Methods

- [x] bert
- [ ] Double bert + LSTM (Double bert model query and reply respectively, then a BiLSTM model the relation between them)
- [ ] LCF [[link]](https://www.mdpi.com/2076-3417/9/16/3389)
- [ ] Semantic Role Labeling

#### Pre-training language models

- [x] ERNIE 1.0
- [x] RoBERTa
- [ ] Pre-training with QA data in the real estate field

#### Data processing

- [x] Truncation
- [x] Clean
- [ ] Shuffle = False
- [ ] Kfold
- [x] Stratified-KFold
- [x] Exchange query-reply pair order
- [ ] **How to utilize duplicate query?**

#### Papers

- [ ] SemBERT [[link]](https://arxiv.org/abs/1909.02209)
- [ ] Explicit Contextual Semantics for Text Comprehension [[link]](https://arxiv.org/abs/1809.02794)



-----------------------------

### Submission history

1. 0.77308091 | bert_spc | ERNIE | Search_f1 | StratifiedKFold 5-fold voting | **BCE loss**

   `python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE`

2. 0.77179287015 | bert_spc | ERNIE | Search_f1 | StratifiedKFold 5-fold voting | **GHMC loss**

   `python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE --criterion ghmc`

3. 0.77587103484 | bert_spc | ERNIE | Search_f1 | StratifiedKFold 5-fold voting | BCE loss | **datareverse**

   `python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 3 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE --datareverse`

4. 0.77975341 | bert_spc | ERNIE | Search_f1 | StratifiedKFold 5-fold voting | BCE loss | **FGM**

   `python train.py --model_name bert_spc --seed 1000 --bert_lr 2e-5 --num_epoch 4 --max_length 100 --cuda 3  --notsavemodel --log_step 20 --pretrained_bert_name ./pretrain_models/ERNIE --attack_type fgm --scheduler`

----------------------

### Licence

[MIT](./LICENSE)

