#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : ChnSetiCorp_trainer.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2021/6/30 15:35
@version: 1.0
@desc  : code for ChnSetiCorp task
"""
import argparse
import json
import os
import random
import gensim 
import numpy as np
from functools import partial

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from torch import nn
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset
from transformers import BertModel,BertTokenizer,AdamW, BertConfig, get_linear_schedule_with_warmup, BertForSequenceClassification
from datasets_chinesebert.collate_functions import collate_to_max_length
from utils.random_seed import set_random_seed
import re

set_random_seed(2333)


class RoBERT(pl.LightningModule):

    def __init__(
        self,
        args: argparse.Namespace
    ):
        """Initialize a models, tokenizer and config."""
        super().__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        self.bert_name = args.bert_name
        self.max_len=args.max_length
        
        self.bert_config = BertConfig.from_pretrained(self.bert_name, output_hidden_states=True,num_labels=3)
        self.bert_model = BertModel.from_pretrained(self.bert_name,config=self.bert_config)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_name)
        self.loss_fn = CrossEntropyLoss()
        #self.acc = pl.metrics.Accuracy(num_classes=self.bert_config.num_labels) #pytorch-lightning 0.9.0v
        self.acc = pl.metrics.Accuracy()#0729因为ddp而修改，适用于pytorch-lightning 1.0.0v        
        gpus_string = self.args.gpus if not self.args.gpus.endswith(',') else self.args.gpus[:-1]
        self.num_gpus = len(gpus_string.split(","))
        self.num_heads=2
        self.dropout=0.3
        self.char_size=768
        
        self.word2id={'unk':0}
        with open(os.path.join(args.charvec_path,'vocab_advgraph_noweight_109706.txt'), encoding='utf-8') as f:
            for line in f.readlines():
                word=line.strip()[0]
                if word not in self.word2id.keys():
                        self.word2id[word] = len(self.word2id)
       
        self.n_words = max(self.word2id.values()) + 1
        self.gensim_model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(args.charvec_path,'word2vec_advgraph_noweight_109706.txt'), binary=False, unicode_errors="ignore")
        self.word_vecs = np.array(np.random.uniform(-1., 1., [self.n_words, self.gensim_model.vector_size]))
        for word in self.word2id.keys():
            try:
                self.word_vecs[self.word2id[word]] = self.gensim_model[word]
            except KeyError:
                pass

        self.char_embeddings = nn.Embedding(self.n_words,self.gensim_model.vector_size)
        self.char_embeddings.weight.data.copy_(torch.from_numpy(self.word_vecs))
        self.char_embeddings.weight.requires_grad = False

        self.encoder_layer=nn.TransformerEncoderLayer(d_model=self.gensim_model.vector_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.map_fc = nn.Linear(self.gensim_model.vector_size*self.max_len, self.char_size)

        self.encoder_layer2=nn.TransformerEncoderLayer(d_model=(self.bert_config.hidden_size+self.char_size), nhead=2,dim_feedforward=256,activation='gelu',dropout=0.3)
        self.transformer_encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=1)

        self.dropout = nn.Dropout(self.dropout)
        self.classifier = nn.Linear((self.bert_config.hidden_size+self.char_size)*(self.max_len), self.bert_config.num_labels)

    def encode_fn(self,sent1,sent2):
  #将text_list embedding成bert模型可用的输入形式
        tokenizer = self.tokenizer(
            sent1,sent2,
            padding = 'max_length',
            truncation = True,
            max_length = self.max_len,
            return_tensors='pt'  # 返回的类型为pytorch tensor
        )
        input_ids = tokenizer['input_ids']
        token_type_ids = tokenizer['token_type_ids']
        attention_mask = tokenizer['attention_mask']
        return input_ids,token_type_ids,attention_mask
    

    def removePunctuation(self,text):
        punctuation = "。，！～？/“”‘’：；【】、「」~!@#$%^&*()_+`{}|\[\]\:\";\-\\\='<>?,./"
        text = re.sub(r'[{}]+'.format(punctuation), '', text)
        return text.strip().lower()


    def node2vec(self,text_list):
        node_vec=[]
        for text in text_list:
            text=self.removePunctuation(text)
            content = [self.word2id.get(w, 0) for w in text]
            content = content[:self.max_len]
            if len(content) < self.max_len:
                content += [self.word2id['unk']] * (self.max_len - len(content))
            node_vec.append(content)
        return node_vec

    def process_data(self, sent1,sent2,text_list, labels):
        input_ids,token_type_ids,attention_mask = self.encode_fn(sent1,sent2)
        node_vec=self.node2vec(text_list)
        node_vec=torch.tensor(node_vec)
        labels = torch.tensor(labels)
        data = TensorDataset(input_ids,token_type_ids,attention_mask,node_vec,labels)
        return data

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          betas=(0.9, 0.98),  # according to RoBERTa paper
                          lr=self.args.lr,
                          eps=self.args.adam_epsilon)
        t_total = len(self.train_dataloader()) // self.args.accumulate_grad_batches * self.args.max_epochs
        warmup_steps = int(self.args.warmup_proporation * t_total)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, token_type_ids,attention_mask,node_vec):
        """"""
        #将输入过滤掉标点之后用node2vec的embedding输出后（max_len,dim），embedding的参数不更新，接全连接层之后，和bert预训练输出的的句子表示(pool_output)做一个concat再接分类器，代码大致写完了，待调试
        #node_vec进embedding输出接全连接再和model的output concat之后接分类器
        # bert_output=self.bert_model(input_ids, token_type_ids, attention_mask).pooler_output
        # node2vec_output=self.char_embeddings(node_vec)
        # attn_output,_=self.attn1(node2vec_output,node2vec_output,node2vec_output,need_weights=False)
        # embed = attn_output.add(node2vec_output)
        # attn_output, _ = self.attn2(embed, embed, embed, need_weights=False)
        # attn_output = node2vec_output.add(attn_output)
        # #print("node2vec_output",node2vec_output.shape)
        # flatten_output=torch.flatten(attn_output,start_dim=1, end_dim=-1) #(batch,max_len*dim)
        # #print("flatten_output",flatten_output.shape)
        # char_output=self.map_fc(flatten_output)#(batch,768)
        # #print("char_output",char_output.shape)
        # #print("bert_output",bert_output.shape)
        # concat_output = torch.cat((char_output, bert_output), 1)#(batch,)
        # #print("concat_output",concat_output.shape)

        # sequence_output = self.dropout(concat_output)
        # logits = self.classifier(sequence_output)
        # #print("logits",logits.shape)

        #bert_output=torch.stack(list(self.bert_model(input_ids, token_type_ids, attention_mask).hidden_states), dim=0)
        bert_output=self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).last_hidden_state
        s=bert_output.shape
        # bert_output=torch.cat((bert_output,torch.zeros(s[0],3,s[2]).cuda()),1)
        #print("bert_output",bert_output.shape)
        # bert_len=s[1]
        # print("bert_len",bert_len)
        # if bert_len<self.max_len:
        #     #bert_output=torch.cat((bert_output,torch.zeros(s[0],self.max_len-bert_len,s[2]).cuda()))
        #     bert_output=bert_output.resize_(s[0],self.max_len,s[2])
        node2vec_output=self.char_embeddings(node_vec)
        encoder_output=self.transformer_encoder(node2vec_output)
        # lstm1_output,_ = self.blstm1(node2vec_output)
        # attn_output, _ = self.attn1(lstm1_output, lstm1_output, lstm1_output, need_weights=False)
        #lstm2_output,_ = self.blstm2(attn_output)
        #print("node2vec_output",node2vec_output.shape)
        flatten_output=torch.flatten(encoder_output,start_dim=1, end_dim=-1) #(batch,max_len*dim)
        #print("flatten_output",flatten_output.shape)
        char_output=self.map_fc(flatten_output)#(batch,768)
        # print("char_output",char_output.shape)
        # print("bert_output",bert_output.shape)
        repeat_output=char_output.reshape(-1,1,self.char_size).repeat(1,self.max_len,1)
        # print("max_len",self.max_len)
        # print("repeat_output",repeat_output.shape)
        concat_output = torch.cat((repeat_output, bert_output), 2)#(batch,max_len,hidden_size*2)
        #print("concat_output",concat_output.shape)
        fusion_output=self.transformer_encoder2(concat_output)
        flatten_fusion_output=torch.flatten(fusion_output,start_dim=1, end_dim=-1)
        #print("flatten_fusion_output",flatten_fusion_output.shape)
        sequence_output = self.dropout(flatten_fusion_output)
        self.representation=sequence_output
        logits = self.classifier(sequence_output)
        #print("logits",logits.shape)

        return logits

    def compute_loss_and_acc(self, batch):
        input_ids,token_type_ids,attention_mask,node_vec,labels = batch #加一个表示 繁体转简体
        # batch_size, length = input_ids.shape
        # pinyin_ids = pinyin_ids.view(batch_size, length, 8)
        y = labels.view(-1)
        #print(y)
        y_hat = self.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            node_vec=node_vec
        )
        # compute loss
        loss = self.loss_fn(y_hat, y)
        # compute acc
        predict_scores = F.softmax(y_hat, dim=1)
        predict_labels = torch.argmax(predict_scores, dim=-1)
        acc = self.acc(predict_labels, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        """"""
        loss, acc = self.compute_loss_and_acc(batch)
        # print(self.parameters())
        # for p in self.parameters():
        #     print(p.shape)
        tf_board_logs = {
            "train_loss": loss,
            "train_acc": acc,
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        return {'loss': loss, 'log': tf_board_logs}

    def validation_step(self, batch, batch_idx):
        """"""
        loss, acc = self.compute_loss_and_acc(batch)
        return {'val_loss': loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        """"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean() #/ self.num_gpus
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        print(avg_loss, avg_acc)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("dev")

    def get_dataloader(self, prefix="train") -> DataLoader:
        """get training dataloader"""

        data_path=os.path.join(self.args.data_dir, prefix + '.tsv')
        sent1=[]
        sent2=[]
        text_list=[]
        labels=[]
        with open(data_path,'r',encoding='utf8') as data_file:
            lines=data_file.readlines()
            lines=lines[1:]
            for line in lines:
                # first, second, third = line.strip().split('\t', 2)
                # sent1.append(first)
                # sent2.append(second)
                # label=label_map[third]
                label, senten_ori= line.strip().split('\t')
                senten=senten_ori.split("[sep]")
                sent1.append(senten[0])
                sent2.append(senten[1])
                text_list.append(senten_ori)
                #print(label)
                
                labels.append(int(label))
        
        dataset_processed=self.process_data(sent1,sent2,text_list,labels)
        dataloader = DataLoader(
            dataset=dataset_processed,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            #collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0]),
            drop_last=False
        )
        return dataloader

    def test_dataloader(self):
        return self.get_dataloader("test")

    def test_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_acc(batch)
        return {'test_loss': loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc = torch.stack([x['test_acc'] for x in outputs]).mean() / self.num_gpus
        tensorboard_logs = {'test_loss': test_loss, 'test_acc': test_acc}
        print(test_loss, test_acc)
        return {'test_loss': test_loss, 'log': tensorboard_logs}


def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_name", required=True, type=str, help="bert config file")
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument("--save_path", required=True, type=str, help="train data path")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=3, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--use_memory", action="store_true", help="load datasets to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of datasets")
    parser.add_argument("--checkpoint_path", type=str, help="train checkpoint")
    parser.add_argument("--save_topk", default=1, type=int, help="save topk checkpoint")
    parser.add_argument("--mode", default='train', type=str, help="train or evaluate")
    parser.add_argument("--warmup_proporation", default=0.01, type=float, help="warmup proporation")
    parser.add_argument("--charvec_path", default='res/char_vec', type=str, help="node2vec data path")

    return parser


def main():
    """main"""
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # create save path if doesn't exit
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    os.makedirs(os.path.join(args.save_path, "checkpoint"), exist_ok=True)

    model = RoBERT(args)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_path, 'checkpoint', '{epoch}-{val_loss:.4f}-{val_acc:.4f}'),
        save_top_k=args.save_topk,
        save_last=False,
        monitor="val_acc",
        mode="max",
    )
    logger = TensorBoardLogger(
        save_dir=args.save_path,
        name='log'
    )

    # save args
    with open(os.path.join(args.save_path, 'checkpoint', "args_node2vec_attn.json"), 'w') as f:
        args_dict = args.__dict__
        del args_dict['tpu_cores']
        json.dump(args_dict, f, indent=4)

    trainer = Trainer.from_argparse_args(args,
                                         checkpoint_callback=checkpoint_callback,
                                         distributed_backend="ddp",
                                         logger=logger)

    trainer.fit(model)

    # print
    result = trainer.test()
    print(result)


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()
