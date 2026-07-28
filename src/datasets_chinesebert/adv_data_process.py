import os
import pandas as pd

path='../adv_result_bertwwmnoext/pwws/'
save_path='../train_datasets/DMSC_adv/'
if os.path.exists(path):
    train_data=[]
    with open(os.path.join(path, 'bert_wwm_noext_dmsc_transformer10_train.txt'), encoding='utf8') as fin:
        flines=fin.readlines()
        for l_ in flines:
            line=l_.split(' ',2)
            if float(line[0])<=0.55:
                train_data.append([int(line[1]),line[2].strip()])
    train_data_ori=pd.read_csv('../train_datasets/DMSC_processed/train.tsv',sep='\t')
    print(len(train_data_ori))


    train_data=pd.DataFrame(train_data)
    print(train_data.shape[0],train_data.shape[1],len(train_data))
    train_data.columns = ['label','text_a']
    train_data=pd.concat([train_data,train_data_ori])
    print(train_data)
    train_data.to_csv(save_path+'train.tsv',sep='\t',index=False)
