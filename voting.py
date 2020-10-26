import os
import pandas as pd
import numpy as np
import copy

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

if __name__ == "__main__":
    kfold_path = './results/bert_spc-cuda-3/kfold/'
    vote(kfold_path)