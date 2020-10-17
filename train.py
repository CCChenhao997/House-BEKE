import os
from pprint import pprint
import torch
import torch.nn as nn
import argparse
import random
import math
import logging
import sys
import copy
import time
from time import strftime, localtime
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from lossfunc.focalloss import FocalLoss, FocalLossBCE
from lossfunc.ghmc import GHMC
from attack import FGM, PGD
from data_utils import Tokenizer4Bert, BertSentenceDataset, get_time_dif
from sklearn.model_selection import StratifiedKFold, KFold
from config import opt, logger, dataset_files


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def search_f1(y_true, y_pred):
    best = 0
    best_t = 0
    for i in range(20, 60):
        tres = i / 100
        y_pred_bin =  (y_pred >= tres)
        score = metrics.f1_score(y_true, y_pred_bin, average='binary')
        if score > best:
            best = score
            best_t = tres
    return best, best_t


class Instructor:
    ''' Model training and evaluation '''
    def __init__(self, test_query, test_reply):
        self.tokenizer = Tokenizer4Bert(opt.max_length, opt.pretrained_bert_name)
        bert_model = BertModel.from_pretrained(opt.pretrained_bert_name, output_hidden_states=True)
        self.model = opt.model_class(bert_model, opt).to(opt.device)

        # * testset
        df_test_query = pd.read_csv(test_query, sep='\t', header=None, encoding='utf-8', engine='python')
        df_test_query.columns=['id','q1']
        df_test_reply = pd.read_csv(test_reply, sep='\t', header=None, encoding='utf-8', engine='python')
        df_test_reply.columns=['id','id_sub','q2']
        df_test_reply['q2'] = df_test_reply['q2'].fillna('好的')
        df_test_data = df_test_query.merge(df_test_reply, how='left')
        df_test_data_reverse = copy.deepcopy(df_test_data[['id', 'q2', 'id_sub', 'q1']])
        self.submit = copy.deepcopy(df_test_reply)

        testset = BertSentenceDataset(df_test_data, self.tokenizer, test=True)
        testset_reverse = BertSentenceDataset(df_test_data_reverse, self.tokenizer, test=True)
        self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.eval_batch_size, shuffle=False)
        self.test_dataloader_reverse = DataLoader(dataset=testset_reverse, batch_size=opt.eval_batch_size, shuffle=False)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(opt.device.index)))
        self._print_args()
    
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))    # 计算参数量 torch.prod - Returns the product of all elements in the input tensor.
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:')
        
        for arg in vars(opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(opt, arg)))
    

    def get_bert_optimizer(self, opt, model):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        diff_part = ["bert.embeddings", "bert.encoder"]

        if opt.diff_lr:
            logger.info("layered learning rate on")
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if
                            not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": opt.weight_decay,
                    "lr": opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                            any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                            not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": opt.weight_decay,
                    "lr": opt.layers_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                            any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": opt.layers_lr
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, eps=opt.adam_epsilon)

        else:
            logger.info("bert learning rate on")
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': opt.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters,
                        lr=opt.bert_lr, eps=opt.adam_epsilon)   #  weight_decay=opt.l2reg

        return optimizer
    
    def _train(self, model, df_train_data, df_dev_data):
        trainset = BertSentenceDataset(df_train_data, self.tokenizer)
        devset = BertSentenceDataset(df_dev_data, self.tokenizer)
        train_dataloader = DataLoader(dataset=trainset, batch_size=opt.train_batch_size, shuffle=True)
        dev_dataloader = DataLoader(dataset=devset, batch_size=opt.eval_batch_size, shuffle=False)

        # 对抗训练
        if opt.adv_type == 'fgm':
            logger.info('对抗选择：fgm')
            fgm = FGM(model)
        elif opt.adv_type == 'pgd':
            logger.info('对抗选择：pgd')
            pgd = PGD(model)
            K = 3

        if opt.criterion == 'focalloss':
            logger.info('criterion选择：focalloss')
            # criterion = FocalLoss(num_class=opt.label_dim, alpha=opt.alpha, gamma=opt.gamma, smooth=opt.smooth)
            criterion = FocalLossBCE(alpha=opt.alpha, gamma=opt.gamma, logits=True)
        elif opt.criterion == 'ghmc':
            logger.info('criterion选择：ghmc')
            criterion = GHMC()
        else:
            logger.info('criterion选择：BCEWithLogitsLoss')
            criterion = nn.BCEWithLogitsLoss()

        optimizer = self.get_bert_optimizer(opt, model)

        if opt.scheduler:
            logger.info('使用scheduler')
            num_training_steps = len(train_dataloader) * opt.num_epoch
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_training_steps*0.1, num_training_steps=num_training_steps)

        max_f1, global_step = 0, 0
        self.best_model = None
        for epoch in range(opt.num_epoch):
            logger.info('>' * 60)
            logger.info('epoch: {}'.format(epoch))
            best_threshold = 0
            targets_all, outputs_all = [], []
            for i_batch, sample_batched in enumerate(train_dataloader):
                global_step += 1
                # switch model to training mode, clear gradient accumulators
                model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(opt.device) for col in opt.inputs_cols]
            
                outputs = model(inputs)
                targets = sample_batched['label'].to(opt.device)
                targets = targets.view(-1, 1).float()

                outputs_all.extend(list(np.array(outputs.cpu() >= opt.threshold, dtype='int')))
                targets_all.extend(list(targets.cpu().detach().numpy()))
                
                loss = criterion(outputs, targets)

                if opt.flooding > 0: # flooding
                    loss = (loss - opt.flooding).abs() + opt.flooding

                loss.backward()

                if opt.adv_type == 'fgm':
                    fgm.attack()  ##对抗训练
                    outputs = model(inputs)
                    loss_adv = criterion(outputs, targets)
                    loss_adv.backward()
                    fgm.restore()

                if opt.adv_type == 'pgd':
                    pgd.backup_grad()
                    for t in range(K):
                        pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                        if t != K-1:
                            model.zero_grad()
                        else:
                            pgd.restore_grad()
                        outputs = model(inputs)
                        loss_adv = criterion(outputs, targets)
                        loss_adv.backward()              # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    pgd.restore()                        # 恢复embedding参数
                
                # torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
                optimizer.step()
                if opt.scheduler:
                    scheduler.step()

                if global_step % opt.log_step == 0:    # 每隔opt.log_step就输出日志
                    train_acc = metrics.accuracy_score(targets_all, outputs_all)
                    test_acc, f1, threshold = self._evaluate(model, dev_dataloader)

                    if f1 > max_f1:
                        best_threshold = threshold
                        max_f1 = f1
                        if not os.path.exists('state_dict'):
                            os.mkdir('state_dict')
                        model_path = './state_dict/{0}_f1_{1:.4f}'.format(opt.model_name, f1)
                        logger.info('>> The {0} has been promoted on {1} with f1 {2:.4f}'.format(opt.model_name, opt.dataset, f1))
                        self.best_model = copy.deepcopy(model)

                    logger.info('step: {}, loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}, threshold:{:.2f}'\
                                .format(i_batch + 1, loss.item(), train_acc, test_acc, f1, threshold))

        logger.info('#' * 100)
        self._evaluate(self.best_model, dev_dataloader, show_results=True)
        return max_f1, model_path, best_threshold
    
    def _evaluate(self, model, dev_dataloader, show_results=False):
        # switch model to evaluation mode
        model.eval()
        targets_all, outputs_all = None, None
        with torch.no_grad():
            for batch, sample_batched in enumerate(dev_dataloader):
                inputs = [sample_batched[col].to(opt.device) for col in opt.inputs_cols]
                targets = sample_batched['label'].to(opt.device)
                outputs = model(inputs)
                
                targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs

        labels = targets_all.data.cpu()
        predict_prob = outputs_all.data.cpu().squeeze()

        best_f1, threshold = search_f1(labels, predict_prob)

        # predic = (outputs_all.cpu() >= opt.threshold)
        predic = (outputs_all.cpu() >= threshold)
        predic = predic.squeeze().long()

        # f1 = metrics.f1_score(labels, predic, average='binary')
        test_acc = metrics.accuracy_score(labels, predic)
        
        if show_results:
            report = metrics.classification_report(labels, predic, digits=4)
            confusion = metrics.confusion_matrix(labels, predic)

            logger.info("Precision, Recall and F1-Score...")
            logger.info(report)
            logger.info("Confusion Matrix...")
            logger.info(confusion)
            logger.info('f1: {:.4f}'.format(best_f1))
            # return report, confusion, f1
            return None

        return test_acc, best_f1, threshold

    
    def _predict(self, dataset, model, best_threshold, max_f1, kfold, reverse=False):
        model.eval()
        targets_all, outputs_all = None, None
        with torch.no_grad():
            for batch, sample_batched in enumerate(dataset):
                inputs = [sample_batched[col].to(opt.device) for col in opt.inputs_cols]
                outputs = model(inputs)
                outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs

        predict = (outputs_all.cpu() >= best_threshold)
        predict = predict.squeeze().long().tolist()

        # 'id','id_sub','q2'
        self.submit['label'] = pd.DataFrame(predict)
        
        DATA_DIR = './results/{}/kfold'.format(opt.model_name)
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR, mode=0o777)
        
        if reverse:
            save_path = DATA_DIR + '/{}-reverse-fold-f1_{:.4f}-{}.tsv'.format(kfold, max_f1, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
        else:
            save_path = DATA_DIR + '/{}-fold-f1_{:.4f}-{}.tsv'.format(kfold, max_f1, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
        self.submit.to_csv(save_path, columns=['id', 'id_sub', 'label'], index=False, header=False, sep='\t')
        logger.info("预测成功！")
    
    def run(self, query_path, reply_path):
        # * trainset & devset
        df_query = pd.read_csv(query_path, sep='\t', header=None, encoding='utf-8', engine='python')
        df_query.columns=['id','q1']
        df_reply = pd.read_csv(reply_path, sep='\t', header=None, encoding='utf-8', engine='python')
        df_reply.columns=['id','id_sub','q2','label']
        df_reply['q2'] = df_reply['q2'].fillna('好的')
        df_data = df_query.merge(df_reply, how='left')
        df_data = df_data[['id', 'q1', 'id_sub', 'q2', 'label']]
        X = np.array(df_data.index)
        y = df_data.loc[:, 'label'].to_numpy()

        skf = StratifiedKFold(n_splits=opt.cross_val_fold, shuffle=True, random_state=opt.seed)
        for kfold, (train_index, dev_index) in enumerate(skf.split(X, y)):
            logger.info("kfold: {}".format(kfold + 1))
            df_train_data = df_data.iloc[train_index]
            df_dev_data = df_data.iloc[dev_index]

            model = copy.deepcopy(self.model)
            max_f1, model_path, best_threshold = self._train(model, df_train_data, df_dev_data)
            logger.info('max_f1: {:.4f}'.format(max_f1))

            if opt.notsavemodel:
                txt_path = model_path + "kfold-{}-".format(kfold+1) + strftime("%Y-%m-%d_%H:%M:%S", localtime()) + '.txt'
                os.mknod(txt_path)
            else:
                torch.save(self.best_model.state_dict(), model_path)
            logger.info('>> saved: {}'.format(model_path))
            self._predict(self.test_dataloader, self.best_model, best_threshold, max_f1, kfold + 1)
            self._predict(self.test_dataloader_reverse, self.best_model, best_threshold, max_f1, kfold + 1, reverse=True)
            logger.info('=' * 60)


def main():
    
    opt.dataset_file = dataset_files[opt.dataset]
    
    # set random seed
    setup_seed(opt.seed)

    if not os.path.exists('log'):
        os.mkdir('log')
    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % ('./log', log_file)))

    start_time = time.time()
    ins = Instructor(opt.dataset_file['test_query'], opt.dataset_file['test_reply'])
    ins.run(opt.dataset_file['train_query'], opt.dataset_file['train_reply'])
    time_dif = get_time_dif(start_time)
    logger.info("Time usage: {}".format(time_dif))

if __name__ == '__main__':
    main()