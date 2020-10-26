import re
import time
from datetime import timedelta
import numpy as np
import pandas as pd
from config import logger, opt
from transformers import BertTokenizer
from torch.utils.data import Dataset

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def parse_data(df_data, test=False):

    # 训练集中reply去重
    if not test:
        df_data = df_data.groupby('id', as_index=False).apply(lambda df: df.drop_duplicates('q2', keep='first'))

    all_data = []
    pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    # id    q1  id_sub	q2	label
    for idx, line in df_data.iterrows():
        try:
            query_id = line[0]
            query = line[1].strip()
            # 去除url和空格
            query = re.sub(pattern, '', query)
            query = query.replace(' ', '')
            query_id_sub = line[2]
            reply = line[3].strip()
            reply = re.sub(pattern, '', reply)
            reply = reply.replace(' ', '')

            # 去除emoji
            query = re.sub(u'[\U00010000-\U0010ffff]', '', query)
            reply = re.sub(u'[\U00010000-\U0010ffff]', '', reply)
            if not re.search('[\w\u4E00-\u9FA5]+', query) or not re.search('[\w\u4E00-\u9FA5]+', reply):
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

        # 多任务，query-reply顺序预测
        if opt.order_predict:
            if idx % 2 == 0:
                data = {'query_id': query_id, 'query': query, 'reply': reply, 'label': label}
                data['order'] = 1
            else:
                data = {'query_id': query_id, 'query': reply, 'reply': query, 'label': label}
                data['order'] = 0
        else:
            data = {'query_id': query_id, 'query': query, 'reply': reply, 'label': label}
        
        all_data.append(data)

        # 数据增强: query-reply逆序
        if not test and opt.datareverse:
            data_reverse = {'query_id': query_id, 'query': reply, 'reply': query, 'label': label}
            all_data.append(data_reverse)

    return all_data


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
        parse = parse_data
        for obj in parse(df_data, test):
            dialogue_pair_indices = tokenizer.text_to_sequence("[CLS] " + obj['query'] + " [SEP] " + obj['reply'] + " [SEP]")
            query_indices = tokenizer.text_to_sequence(obj['query'])
            reply_indices = tokenizer.text_to_sequence(obj['reply'])
            bert_segments_ids = np.asarray([0] * (np.sum(query_indices != 0) + 2) + [1] * (np.sum(reply_indices != 0) + 1))
            bert_segments_ids = tokenizer.pad_sequence(bert_segments_ids, 0, tokenizer.max_length)
            attention_mask = np.asarray([1] * np.sum(dialogue_pair_indices != 0) + [0] * (opt.max_length - np.sum(dialogue_pair_indices != 0)))
            label = obj['label']
            sub_data = {
                    'dialogue_pair_indices': dialogue_pair_indices,
                    'bert_segments_ids': bert_segments_ids,
                    'attention_mask': attention_mask,
                    'label': label,
                }

            if opt.order_predict:
                sub_data['order'] = obj['order']

            data.append(sub_data)

        self._data = data
    
    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)


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
    data = parse_data(df_data)
    print(data[0])
