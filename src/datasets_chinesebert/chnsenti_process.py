from zhconv import convert
import pandas as pd
import os

file=['train.tsv','dev.tsv','test.tsv']
path='../train_datasets/ChnSetiCorp'
save_path='../train_datasets/ChnSetiCorp_process/'
for f in file:
    df = pd.read_csv(os.path.join(path,f), sep='\t')
    print(df)
    rows=[]
    #print(df[1])
    for index, row in df.iterrows():
        #print(row)
        text=convert(row['text_a'],'zh-hans')
        rows.append([row['label'],text])
    rows=pd.DataFrame(rows)
    #print(rows.shape[0],rows.shape[1],len(rows))
    rows.columns = ['label','text_a']
    print(rows)
    rows.to_csv(save_path+f,sep='\t',index=False)
