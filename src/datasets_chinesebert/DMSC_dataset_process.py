import pandas as pd
import random
import os

path='../train_datasets/DMSC.csv'
save_path='../train_datasets/DMSC_processed/'
raw_data = pd.read_csv(path)
print("the length of dataset:", len(raw_data))
# print("the distribution of dataset:")
# print(raw_data['Star'].value_counts())

def polarity (row):
  if row['Star'] >= 4: # bigger or equal to 4 stars are positive
    return '1'
  if row['Star'] <= 3: # smaller or equal to 3 stars are negative
    return '0'

def stripe_comment (row):
    return row['Comment'].replace('\n',' ')

raw_data['Star'] = raw_data.apply(lambda row: polarity(row), axis=1)
raw_data['Comment'] = raw_data.apply(lambda row: stripe_comment(row), axis=1)
print(raw_data['Star'].value_counts())
processed_data = raw_data[['Star','Comment']]
print(processed_data.head())

n_total = len(processed_data)

n_sample=int(n_total * 0.01)

processed_data=processed_data[:n_sample]
print(n_total)
offset = int(n_sample * 0.8)
processed_data=processed_data.sample(frac=1).reset_index(drop=True)
train_set = processed_data[:offset]
test_set = processed_data[offset:]
print(train_set.head())
print(test_set.head())
print("train:",train_set['Star'].value_counts())
print("test:",test_set['Star'].value_counts())

train_set.columns = ['label','text_a']
print(train_set)
test_set.columns = ['label','text_a']
print(test_set)

if not os.path.exists(save_path):
    os.mkdir(save_path)
train_set.to_csv(save_path+'train.tsv',sep='\t',index=False)
test_set.to_csv(save_path+'test.tsv',sep='\t',index=False)
test_set.to_csv(save_path+'dev.tsv',sep='\t',index=False)
