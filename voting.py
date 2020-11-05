import os
import pandas as pd
import numpy as np
import copy
from config import opt, dataset_files
from pprint import pprint

def work(kfold):
    count = [0, 0]
    for i in kfold:
        count[i] += 1
    out = count.index(max(count))
    return out

def vote(kfold_path):
    files = os.listdir(kfold_path)
    files = [i for i in files]

    i = 0
    df_merged = None
    for fname in files:
        tmp_df = pd.read_csv(kfold_path + fname, sep='\t', header=None)
        tmp_df.columns = ['id','id_sub','label']
        # tmp_df_left = copy.deepcopy(tmp_df[['id', 'label']])
        # pprint(tmp_df.head(10))
        if i == 0:
            df_merged = pd.read_csv(kfold_path + fname, sep='\t', header=None)
            df_merged.columns = ['id','id_sub','label']
        if i > 0:
            df_merged = df_merged.merge(tmp_df, how='left', on=['id', 'id_sub'])
        i += 1

    tmp_label = np.array(df_merged.iloc[:, 2:])
    voted_label = [work(line) for line in tmp_label]
    df_summit = copy.deepcopy(df_merged[['id', 'id_sub']])
    df_summit['label'] = voted_label

    df_summit.to_csv(kfold_path + 'voted.tsv', index=False, header=False, sep='\t')
    print("Vote successful!")


def generate_pseudo_data(test_query, test_reply, kfold_path):
    # test_dataset
    df_test_query = pd.read_csv(test_query, sep='\t', header=None, encoding='utf-8', engine='python')
    df_test_query.columns=['id','q1']
    df_test_reply = pd.read_csv(test_reply, sep='\t', header=None, encoding='utf-8', engine='python')
    df_test_reply.columns=['id','id_sub','q2']
    df_test_reply['q2'] = df_test_reply['q2'].fillna('好的')
    df_test_data = df_test_query.merge(df_test_reply, how='left')
    # pprint(df_test_data.head(10))

    # kfold
    files = os.listdir(kfold_path)
    files = [i for i in files]
    if 'voted.tsv' in files:
        files.remove('voted.tsv')
    K = len(files)
    print("K: ", K)
    
    i = 0
    df_merged = None
    for fname in files:
        tmp_df = pd.read_csv(kfold_path + fname, sep='\t', header=None)
        tmp_df.columns = ['id','id_sub','label']
        # tmp_df_left = copy.deepcopy(tmp_df[['id', 'label']])
        # pprint(tmp_df.head(10))
        if i == 0:
            df_merged = pd.read_csv(kfold_path + fname, sep='\t', header=None)
            df_merged.columns = ['id','id_sub','label']
        if i > 0:
            df_merged = df_merged.merge(tmp_df, how='left', on=['id', 'id_sub'])
        i += 1

    # print("################################################")
    # print(df_merged.head(10))
    pseudo_data = df_test_data.merge(df_merged, how='left', on=['id', 'id_sub'])
    
    pos_ids, pos_q1s, pos_id_subs, pos_q2s, pos_labels = [], [], [], [], []
    neg_ids, neg_q1s, neg_id_subs, neg_q2s, neg_labels = [], [], [], [], []
    for idx, line in pseudo_data.iterrows():
        label_list = []
        label = None

        dia_id = line[0]
        query  = line[1]
        id_sub = line[2]
        reply  = line[3]
        for i in range(K):
            label_list.append(line[i + 4])
        if sum(label_list) == K:
            label = 1
            pos_labels.append(label)
            pos_ids.append(dia_id)
            pos_q1s.append(query)
            pos_id_subs.append(id_sub)
            pos_q2s.append(reply)
        elif sum(label_list) == 0:
            label = 0
            neg_labels.append(label)
            neg_ids.append(dia_id)
            neg_q1s.append(query)
            neg_id_subs.append(id_sub)
            neg_q2s.append(reply)
        
    pos_dict = {
        'id': pos_ids,
        'q1': pos_q1s,
        'id_sub': pos_id_subs,
        'q2': pos_q2s,
        'label': pos_labels
    }

    neg_dict = {
        'id': neg_ids,
        'q1': neg_q1s,
        'id_sub': neg_id_subs,
        'q2': neg_q2s,
        'label': neg_labels
    }
    
    pos_df = pd.DataFrame(pos_dict)
    neg_df = pd.DataFrame(neg_dict)

    # print(pos_df.head(10))
    # print(pos_df.shape)
    # print("####################")
    # print(neg_df.head(10))
    # print(neg_df.shape)
    return pos_df, neg_df


if __name__ == "__main__":
    kfold_path = './results/bert_spc-cuda-2-GHMC-ERNIE-ALL-TAPT-1103/kfold/'
    # vote(kfold_path)

    # opt.dataset_file['test_query'], opt.dataset_file['test_reply']
    opt.dataset_file = dataset_files[opt.dataset]
    generate_pseudo_data(opt.dataset_file['test_query'], opt.dataset_file['test_reply'], kfold_path)