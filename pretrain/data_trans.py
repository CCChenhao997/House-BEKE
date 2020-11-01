#coding=utf-8
import re
import copy
import pandas as pd
from sklearn.utils import shuffle


pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

def trans(query_path, reply_path, gen_path):
    df_query = pd.read_csv(query_path, sep='\t', header=None, encoding='utf-8', engine='python')
    df_query.columns=['id','q1']
    df_query = copy.deepcopy(df_query['q1'])
    df_reply = pd.read_csv(reply_path, sep='\t', header=None, encoding='utf-8', engine='python')
    df_reply.columns=['id','id_sub','q2', 'label']
    df_reply = copy.deepcopy(df_reply['q2'])
    
    frame = [df_query, df_reply]
    df_concat = pd.DataFrame(pd.concat(frame, axis=0))
    df_concat = shuffle(df_concat)

    f = open(gen_path, 'w+')
    max_length = 0
    for idx, line in df_concat.iterrows():
        context = str(line[0]).strip()
        context = re.sub(pattern, '', context)
        context = re.sub(u'[\U00010000-\U0010ffff]', '', context)
        context_len = len(context)
        if max_length < context_len:
            max_length = context_len
        if context_len < 3:
            continue
        context_left  = context[: context_len // 2]
        context_right = context[context_len // 2 :]
        f.write(context_left + '\r\n')
        f.write(context_right + '\r\n')
        f.write('\r\n')
    
    print("max_length={}".format(max_length))
    print("写入完成!")


if __name__ == "__main__":
    # query_path = r'./data/test/test.query.tsv'
    # reply_path = r'./data/test/test.reply.tsv'
    
    query_path = r'./data/train/train.query.tsv'
    reply_path = r'./data/train/train.reply.tsv'
    
    gen_path = r'./pretrain/pretrain_data/beke-pretrain_train.txt'
    trans(query_path, reply_path, gen_path)
