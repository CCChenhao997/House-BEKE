import re
import time
import torch
from datetime import timedelta
import numpy as np
from numpy.core.arrayprint import printoptions
import pandas as pd
from config import logger, opt
from transformers import BertTokenizer
from torch.utils.data import Dataset
from pprint import pprint

pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def parse_data_dialogue(df_data, test=False):
    # id    q1  id_sub	q2	label
    # all_data = []
    all_data = dict()
    
    df_data = df_data.groupby('id', as_index=False)
    for _id, group in df_data:
        for idx, line in group.iterrows():
            query_id = line[0]
            query = line[1].strip()
            query = re.sub(pattern, '链接', query)
            query = re.sub(r'\s+', ' ', query)
            query_id_sub = line[2]
            reply = line[3].strip()
            reply = re.sub(pattern, '链接', reply)
            reply = re.sub(r'\s+', ' ', reply)

            if len(query) == 0 or len(reply) == 0:
                logger.info("query or reply is empty!")
                exit()

            if test:    # 测试集
                label = 0
            else:
                label = line[4]

            # 句子长度截断处理
            while len(query) + len(reply) > opt.max_length:
                if len(query) <= len(reply) and len(query) <= (opt.max_length // 2):
                    reply = reply[: opt.max_length - len(query)]
                elif len(query) > len(reply) and len(reply) <= (opt.max_length // 2):
                    query = query[: opt.max_length - len(reply)]
                else:
                    query = query[: opt.max_length // 2]
                    reply = reply[: opt.max_length // 2]
                
            data = {'query': query, 'reply': reply, 'label': label}
            if query_id not in all_data:
                all_data[query_id] = []
                all_data[query_id].append(data)
            else:
                all_data[query_id].append(data)
    
    return all_data


def parse_data(df_data, test=False):

    # 训练集中reply去重
    if not test and opt.drop_duplicates and not opt.dialogue:
        df_data = df_data.groupby('q1', as_index=False).apply(lambda df: df.drop_duplicates('q2', keep='first'))
        # df_data = df_data.groupby('id', as_index=False).apply(lambda df: df.drop_duplicates('q2', keep=False))

    all_data = []
    # id    q1  id_sub	q2	label
    for idx, line in df_data.iterrows():
        try:
            query_id = line[0]
            query = line[1].strip()
            # 去除url和空格 (多个连续空格转1个空格)
            query = re.sub(pattern, '链接', query)
            # query = query.replace(' ', '')
            query = re.sub(r'\s+', ' ', query)
            query_id_sub = line[2]
            reply = line[3].strip()
            reply = re.sub(pattern, '链接', reply)
            # reply = reply.replace(' ', '')
            reply = re.sub(r'\s+', ' ', reply)

            # 去除emoji
            if not test:
                query = re.sub(u'[\U00010000-\U0010ffff]', '', query)
                reply = re.sub(u'[\U00010000-\U0010ffff]', '', reply)
                # 去掉没有中文的样本
                # if not re.search('[\w\u4E00-\u9FA5]+', query) or not re.search('[\w\u4E00-\u9FA5]+', reply):
                #     continue
                if len(query) == 0 or len(reply) == 0:
                    continue
            
            if len(query) == 0 or len(reply) == 0:
                logger.info("query or reply is empty!")
                exit()

            if test:    # 测试集
                label = 0
            else:
                label = line[4]

            # 句子长度截断处理
            while len(query) + len(reply) > opt.max_length:
                if len(query) <= len(reply) and len(query) <= (opt.max_length // 2):
                    reply = reply[: opt.max_length - len(query)]
                elif len(query) > len(reply) and len(reply) <= (opt.max_length // 2):
                    query = query[: opt.max_length - len(reply)]
                else:
                    query = query[: opt.max_length // 2]
                    reply = reply[: opt.max_length // 2]
  
        except:
            logger.info('{}'.format(line))
            exit()

        data = {'query_id': query_id, 'query': query, 'reply': reply, 'label': label}
        all_data.append(data)

        # 数据增强: query-reply逆序
        if not test and opt.datareverse:
            data_reverse = {'query_id': query_id, 'query': reply, 'reply': query, 'label': label}
            all_data.append(data_reverse)

    logger.info('样本数: {}'.format(len(all_data)))
    return all_data


def case_data(df_data):

    if opt.drop_duplicates:
        df_data = df_data.groupby('q1', as_index=False).apply(lambda df: df.drop_duplicates('q2', keep='first'))
        # df_data = df_data.groupby('id', as_index=False).apply(lambda df: df.drop_duplicates('q2', keep=False))

    query_id_list, query_list, query_id_sub_list, reply_list, label_list = [], [], [], [], []
    pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    # id    q1  id_sub	q2	label
    for idx, line in df_data.iterrows():
        try:
            query_id = line[0]
            query = line[1].strip()
            query = re.sub(pattern, '链接', query)
            query = re.sub(r'\s+', ' ', query)
            query_id_sub = line[2]
            reply = line[3].strip()
            reply = re.sub(pattern, '链接', reply)
            reply = re.sub(r'\s+', ' ', reply)
            label = line[4]

            # 去除emoji
            query = re.sub(u'[\U00010000-\U0010ffff]', '', query)
            reply = re.sub(u'[\U00010000-\U0010ffff]', '', reply)
            if len(query) == 0 or len(reply) == 0:
                continue

            # 句子长度截断处理
            while len(query) + len(reply) > opt.max_length:
                if len(query) <= len(reply) and len(query) <= (opt.max_length // 2):
                    reply = reply[: opt.max_length - len(query)]
                elif len(query) > len(reply) and len(reply) <= (opt.max_length // 2):
                    query = query[: opt.max_length - len(reply)]
                else:
                    query = query[: opt.max_length // 2]
                    reply = reply[: opt.max_length // 2]
  
        except:
            logger.info('{}'.format(line))
            exit()

        query_id_list.append(query_id)
        query_list.append(query)
        query_id_sub_list.append(query_id_sub)
        reply_list.append(reply)
        label_list.append(label)

    return query_id_list, query_list, query_id_sub_list, reply_list, label_list


class Tokenizer4Bert(object):
    def __init__(self, max_length, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_length = max_length

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return Tokenizer4Bert.pad_sequence(sequence, pad_id=0, maxlen=self.max_length, 
                                    padding=padding, truncating=truncating)

    @staticmethod
    def pad_sequence(sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
        # x = (np.zeros(maxlen) + pad_id).astype(dtype)   # 长度为maxlen的数组中的元素全为pad_id，也就是0
        x = (np.ones(maxlen) * pad_id).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]  # 把过长的句子前面部分截断
        else:
            trunc = sequence[:maxlen]   # 把过长的句子尾部截断
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc      # 在句子尾部打padding
        else:
            x[-len(trunc):] = trunc     # 在句子前面打padding
        return x

    @staticmethod
    def split_text(text):
        for ch in ["\'s", "\'ve", "n\'t", "\'re", "\'m", "\'d", "\'ll", ",", ".", "!", "*", "/", "?", "(", ")", "\"", "-", ":"]:
            text = text.replace(ch, " "+ch+" ")
        return text


class BertSentenceDataset(Dataset):
    ''' PyTorch standard dataset class '''
    def __init__(self, df_data, tokenizer, test=False):
        data = list()
        if opt.dialogue:
            parse = parse_data_dialogue
            for key, value in parse(df_data, test).items():
                dialogue_data = []
                dialogue_id = key
                for obj in value:
                    dialogue_pair_indices = tokenizer.text_to_sequence("[CLS] " + obj['query'] + " [SEP] " + obj['reply'] + " [SEP]")
                    query_indices = tokenizer.text_to_sequence(obj['query'])
                    reply_indices = tokenizer.text_to_sequence(obj['reply'])
                    bert_segments_ids = np.asarray([0] * (np.sum(query_indices != 0) + 2) + [1] * (np.sum(reply_indices != 0) + 1))
                    bert_segments_ids = tokenizer.pad_sequence(bert_segments_ids, 0, tokenizer.max_length)
                    attention_mask = np.asarray([1] * np.sum(dialogue_pair_indices != 0) + [0] * (opt.max_length - np.sum(dialogue_pair_indices != 0)))
                    label = obj['label']
                    dialogue_data.append(
                        {
                            'dialogue_pair_indices': dialogue_pair_indices,
                            'bert_segments_ids': bert_segments_ids,
                            'attention_mask': attention_mask,
                            'label': label,
                        }
                    )

                data.append(dialogue_data)

        else:
            parse = parse_data
            for obj in parse(df_data, test):
                dialogue_pair_indices = tokenizer.text_to_sequence("[CLS] " + obj['query'] + " [SEP] " + obj['reply'] + " [SEP]")
                dialogue_pair_indices_reverse = tokenizer.text_to_sequence("[CLS] " + obj['reply'] + " [SEP] " + obj['query'] + " [SEP]")
                query_indices = tokenizer.text_to_sequence(obj['query'])
                reply_indices = tokenizer.text_to_sequence(obj['reply'])
                bert_segments_ids = np.asarray([0] * (np.sum(query_indices != 0) + 2) + [1] * (np.sum(reply_indices != 0) + 1))
                bert_segments_ids = tokenizer.pad_sequence(bert_segments_ids, 0, tokenizer.max_length)
                bert_segments_ids_reverse = np.asarray([0] * (np.sum(reply_indices != 0) + 2) + [1] * (np.sum(query_indices != 0) + 1))
                bert_segments_ids_reverse = tokenizer.pad_sequence(bert_segments_ids_reverse, 0, tokenizer.max_length)
                attention_mask = np.asarray([1] * np.sum(dialogue_pair_indices != 0) + [0] * (opt.max_length - np.sum(dialogue_pair_indices != 0)))
                attention_mask_reverse = np.asarray([1] * np.sum(dialogue_pair_indices_reverse != 0) + [0] * (opt.max_length - np.sum(dialogue_pair_indices_reverse != 0)))
                attention_mask_query = np.asarray([1] * np.sum(query_indices != 0) + [0] * (opt.max_length - np.sum(query_indices != 0)))
                attention_mask_reply = np.asarray([1] * np.sum(reply_indices != 0) + [0] * (opt.max_length - np.sum(reply_indices != 0)))
                label = obj['label']
                sub_data = {
                        'dialogue_pair_indices': dialogue_pair_indices,
                        'dialogue_pair_indices_reverse': dialogue_pair_indices_reverse,
                        'bert_segments_ids': bert_segments_ids,
                        'bert_segments_ids_reverse': bert_segments_ids_reverse,
                        'attention_mask': attention_mask,
                        'attention_mask_reverse': attention_mask_reverse,
                        'query_indices': query_indices,
                        'reply_indices': reply_indices,
                        'attention_mask_query': attention_mask_query,
                        'attention_mask_reply': attention_mask_reply,
                        'label': label,
                    }

                data.append(sub_data)

        self._data = data
    
    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)


def collate_wrapper(batch):
    dialogue_pair_indices = torch.LongTensor([item['dialogue_pair_indices'] for item in batch[0]]).detach()
    bert_segments_ids = torch.LongTensor([item['bert_segments_ids'] for item in batch[0]]).detach()
    attention_mask = torch.LongTensor([item['attention_mask'] for item in batch[0]]).detach()
    label = torch.LongTensor([item['label'] for item in batch[0]]).detach()
    data = {
            'dialogue_pair_indices': dialogue_pair_indices,
            'bert_segments_ids': bert_segments_ids,
            'attention_mask': attention_mask,
            'label': label,
        }
    return data


if __name__ == '__main__':
    query_path = './data/train/train.query.tsv'
    reply_path = './data/train/train.reply.tsv'
    df_query = pd.read_csv(query_path, sep='\t', header=None, encoding='utf-8', engine='python')
    df_query.columns = ['id', 'q1']
    df_reply = pd.read_csv(reply_path, sep='\t', header=None, encoding='utf-8', engine='python')
    df_reply.columns = ['id', 'id_sub', 'q2', 'label']
    df_reply['q2'] = df_reply['q2'].fillna('好的')
    df_data = df_query.merge(df_reply, how='left')
    df_data = df_data[['id', 'q1', 'id_sub', 'q2', 'label']]
    # X = np.array(df_data.index)
    # y = df_data.loc[:, 'label'].to_numpy()
    # data = parse_data(df_data)
    data = parse_data_dialogue(df_data)
    pprint(data)
