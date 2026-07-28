import os
import pandas as pd

path='../train_datasets/review/'
save_path='../train_datasets/review_processed/'
if os.path.exists(path):
    train_data=[]
    with open(os.path.join(path, 'ham_train.txt'), encoding='utf8') as fin:
        flines=fin.readlines()
        for l_ in flines:
            line=l_.replace(' ','').strip()
            train_data.append([1,line])
    with open(os.path.join(path, 'spam_train.txt'), encoding='utf8') as fin:
        flines=fin.readlines()
        for l_ in flines:
            line=l_.replace(' ','').strip()
            train_data.append([0,line])
    train_data=pd.DataFrame(train_data)
    print(train_data.shape[0],train_data.shape[1],len(train_data))
    train_data.columns = ['label','text_a']
    print(train_data)
    train_data.to_csv(save_path+'train.tsv',sep='\t',index=False)

    dev_data=[]
    with open(os.path.join(path, 'ham_test.txt'), encoding='utf8') as fin:
        flines=fin.readlines()
        for l_ in flines:
            line=l_.replace(' ','').strip()
            dev_data.append([1,line])
    with open(os.path.join(path, 'spam_test.txt'), encoding='utf8') as fin:
        flines=fin.readlines()
        for l_ in flines:
            line=l_.replace(' ','').strip()
            dev_data.append([0,line])
    dev_data=pd.DataFrame(dev_data)
    print(dev_data.shape[0],dev_data.shape[1],len(dev_data))
    dev_data.columns = ['label','text_a']
    print(dev_data)
    dev_data.to_csv(save_path+'dev.tsv',sep='\t',index=False)
            