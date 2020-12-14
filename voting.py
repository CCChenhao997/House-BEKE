import os
import pandas as pd
import numpy as np
import copy

from config import opt, dataset_files, logger
from pprint import pprint
from sklearn import metrics

def ratio(label_path):
    df = pd.read_csv(label_path, sep='\t', header=None, encoding='utf-8', engine='python')
    df.columns=['id','id_sub', 'label']
    logger.info("预测标签比例为:")
    logger.info(df['label'].value_counts())

def search_f1(y_true, y_pred):
    best = 0
    best_t = 0
    for i in range(opt.start, opt.end):
        tres = i / 100
        y_pred_bin =  (y_pred >= tres)
        score = metrics.f1_score(y_true, y_pred_bin, average='binary')
        if score > best:
            best = score
            best_t = tres
    return best, best_t

def kfold_search_f1(kfold_path):
    files = os.listdir(kfold_path)
    files = [i for i in files]

    i = 0
    df_merged = None
    for fname in files:
        tmp_df = pd.read_csv(kfold_path + fname, sep='\t')
        if i == 0:
            df_merged = pd.read_csv(kfold_path + fname, sep='\t')
        elif i > 0:
            df_merged = df_merged.append(tmp_df, sort=False)
    
    # print(df_merged.head(10))
    y_true = df_merged['label']
    y_pred = df_merged['softlabel']
    f1_score, threshold = search_f1(y_true, y_pred)
    logger.info("f1_score:{:.4f}, threshold:{:.2f}".format(f1_score, threshold))
    return f1_score, threshold

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

    df_summit.to_csv(kfold_path + 'vote.tsv', index=False, header=False, sep='\t')
    print("Vote successful!")


def kfold_result_combined(kfold_path, pattern='vote', threshold=0.5):
    files_name = os.listdir(kfold_path)
    files_path = [os.path.join(kfold_path, fname) for fname in files_name]
    df_merged = None
    weight = []
    for idx, fname in enumerate(files_path):
        if files_name[idx] in ['weighted.tsv', 'vote.tsv', 'average.tsv']:
            continue
        tmp_df = pd.read_csv(fname, sep='\t', header=None)
        tmp_df.columns = ['id','id_sub','label']
        weight.append(float(files_name[idx].split('-')[2].split('_')[1])) 
        if df_merged is None:
            df_merged = copy.deepcopy(tmp_df)
            df_merged.columns = ['id','id_sub','label']
        else:
            df_merged = df_merged.merge(tmp_df, how='left', on=['id', 'id_sub'])
    tmp_label = df_merged.iloc[:, 2:].to_numpy()

    def average_result(all_result):  # shape:[num_model, axis]
        all_result = np.asarray(all_result, dtype=np.float)
        return np.mean(all_result, axis=1)

    def weighted_result(all_result, weight):
        all_result = np.asarray(all_result, dtype=np.float)
        return np.average(all_result, axis=1, weights=weight)

    def vote_result(all_result):
        all_result = np.asarray(all_result, dtype=np.int)
        lens = (all_result.shape[1] + 1) // 2
        all_result = np.sum(all_result, axis=1)
        return [1 if ct>=lens else 0 for ct in all_result]

    def threshold_split(result_data, threshold=0.5):
        return list(np.array(result_data >= threshold, dtype='int'))

    combined_result = {
        'vote': lambda : vote_result(tmp_label),
        'weighted': lambda : threshold_split(weighted_result(tmp_label, weight=weight), threshold),
        'average': lambda : threshold_split(average_result(tmp_label), threshold), 
    }[pattern]()

    df_summit = copy.deepcopy(df_merged[['id', 'id_sub']])
    df_summit['label'] = combined_result
    
    df_summit.to_csv(kfold_path + pattern + '.tsv', index=False, header=False, sep='\t')
    print("{} successful!".format(pattern))


def generate_pseudo_data(test_query, test_reply, kfold_path):
    # test_dataset
    df_test_query = pd.read_csv(test_query, sep='\t', header=None, encoding='utf-8', engine='python')
    df_test_query.columns=['id','q1']
    df_test_reply = pd.read_csv(test_reply, sep='\t', header=None, encoding='utf-8', engine='python')
    df_test_reply.columns=['id','id_sub','q2']
    df_test_reply['q2'] = df_test_reply['q2'].fillna('好的')
    df_test_data = df_test_query.merge(df_test_reply, how='left')

    # kfold
    files = os.listdir(kfold_path)
    files = [i for i in files]
    remove_list = ['weighted.tsv', 'voted.tsv', 'vote.tsv', 'average.tsv']
    for re in remove_list:
        if re in files:
            files.remove(re)

    K = len(files)
    print("K: ", K)
    # assert K == opt.cross_val_fold, "K not eq cross_val_fold."
    
    i = 0
    df_merged = None
    for fname in files:
        tmp_df = pd.read_csv(kfold_path + fname, sep='\t', header=None)
        tmp_df.columns = ['id','id_sub','label']
        if i == 0:
            df_merged = pd.read_csv(kfold_path + fname, sep='\t', header=None)
            df_merged.columns = ['id','id_sub','label']
        if i > 0:
            df_merged = df_merged.merge(tmp_df, how='left', on=['id', 'id_sub'])
        i += 1

    pseudo_data = df_test_data.merge(df_merged, how='left', on=['id', 'id_sub'])
    
    easy_pos_ids, easy_pos_q1s, easy_pos_id_subs, easy_pos_q2s, easy_pos_labels = [], [], [], [], []
    easy_neg_ids, easy_neg_q1s, easy_neg_id_subs, easy_neg_q2s, easy_neg_labels = [], [], [], [], []
    hard_pos_ids, hard_pos_q1s, hard_pos_id_subs, hard_pos_q2s, hard_pos_labels = [], [], [], [], []
    hard_neg_ids, hard_neg_q1s, hard_neg_id_subs, hard_neg_q2s, hard_neg_labels = [], [], [], [], []
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
            easy_pos_labels.append(label)
            easy_pos_ids.append(dia_id)
            easy_pos_q1s.append(query)
            easy_pos_id_subs.append(id_sub)
            easy_pos_q2s.append(reply)
        elif sum(label_list) == 0:
            label = 0
            easy_neg_labels.append(label)
            easy_neg_ids.append(dia_id)
            easy_neg_q1s.append(query)
            easy_neg_id_subs.append(id_sub)
            easy_neg_q2s.append(reply)
        elif sum(label_list) >= ((K+1) // 2):
            label = 1
            hard_pos_labels.append(label)
            hard_pos_ids.append(dia_id)
            hard_pos_q1s.append(query)
            hard_pos_id_subs.append(id_sub)
            hard_pos_q2s.append(reply)
        else:
            label = 0
            hard_neg_labels.append(label)
            hard_neg_ids.append(dia_id)
            hard_neg_q1s.append(query)
            hard_neg_id_subs.append(id_sub)
            hard_neg_q2s.append(reply)
        
    easy_pos_dict = {
        'id': easy_pos_ids,
        'q1': easy_pos_q1s,
        'id_sub': easy_pos_id_subs,
        'q2': easy_pos_q2s,
        'label': easy_pos_labels
    }

    easy_neg_dict = {
        'id': easy_neg_ids,
        'q1': easy_neg_q1s,
        'id_sub': easy_neg_id_subs,
        'q2': easy_neg_q2s,
        'label': easy_neg_labels
    }

    hard_pos_dict = {
        'id': hard_pos_ids,
        'q1': hard_pos_q1s,
        'id_sub': hard_pos_id_subs,
        'q2': hard_pos_q2s,
        'label': hard_pos_labels
    }

    hard_neg_dict = {
        'id': hard_neg_ids,
        'q1': hard_neg_q1s,
        'id_sub': hard_neg_id_subs,
        'q2': hard_neg_q2s,
        'label': hard_neg_labels
    }
    
    easy_pos_df = pd.DataFrame(easy_pos_dict)
    easy_neg_df = pd.DataFrame(easy_neg_dict)
    hard_pos_df = pd.DataFrame(hard_pos_dict)
    hard_neg_df = pd.DataFrame(hard_neg_dict)

    return easy_pos_df, easy_neg_df, hard_pos_df, hard_neg_df


if __name__ == "__main__":
    # kfold_path = './results/bert_spc-cuda-2-GHMC-ERNIE-ALL-TAPT-1103/kfold/'
    kfold_path = './results/fusion-1118/'
    vote(kfold_path)

    # * kfold vote
    # kfold_result_combined(kfold_path, pattern='weighted', threshold=0.50)
    # kfold_result_combined(kfold_path, pattern='vote')
    # kfold_result_combined(kfold_path, pattern='average', threshold=0.5)

    # * generate pseudo data
    # opt.dataset_file = dataset_files[opt.dataset]
    # generate_pseudo_data(opt.dataset_file['test_query'], opt.dataset_file['test_reply'], kfold_path)

    # * find global f1-threshold
    # kfold_search_f1('./results/bert_spc-cuda-2-GHMC-ERNIE-ALL-TAPT-1103/casestudy/')