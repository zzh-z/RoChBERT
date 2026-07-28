import OpenAttack
import datasets
from trainer_inference_chinesebert import XNLITask
import json
import argparse
import torch
import pytorch_lightning as pl
import os
from datasets_chinesebert.xnli_dataset import XNLIDataset
#from pypinyin import pinyin, Style
import numpy as np
import pandas
from OpenAttack.substitutes.chinese_glyph_pron_char import ChineseGlyphPronSubstitute

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChineseBert_model(OpenAttack.Classifier):
    def __init__(self,model,args):
        self.args = args
        self.model=model.to(device)
        
    
    # access to the classification probability scores with respect input sentences
    def get_prob(self, input_):
        ret = []
        for sent in input_:
            if "[sep]" not in sent:
                senten=sent.split("[")
                if len(senten)==1:
                    senten=sent.split("s")
            else:
                senten=sent.split("[sep]")
            
            xnli_process = XNLIDataset(data_path='res/data/OCNLI_processed/test.tsv',chinese_bert_path=self.args.bert_path)
            first_output = xnli_process.tokenizer.encode(senten[0], add_special_tokens=False)
            first_pinyin_tokens = xnli_process.convert_sentence_to_pinyin_ids(senten[0], first_output)
            second_output = xnli_process.tokenizer.encode(senten[1], add_special_tokens=False)
            second_pinyin_tokens = xnli_process.convert_sentence_to_pinyin_ids(senten[1], second_output)
            # label = self.label_map[third]
            # convert sentence to id
            bert_tokens = first_output.ids + [102] + second_output.ids
            pinyin_tokens = first_pinyin_tokens + [[0] * 8] + second_pinyin_tokens
            # if len(bert_tokens) > self.max_length - 2:
            #     bert_tokens = bert_tokens[:self.max_length - 2]
            #     pinyin_tokens = pinyin_tokens[:self.max_length - 2]

            # # assert
            # assert len(bert_tokens) <= self.max_length
            # assert len(bert_tokens) == len(pinyin_tokens)
            # 转化list为tensor
            input_ids = torch.LongTensor([101] + bert_tokens + [102])
            pinyin_ids = torch.LongTensor([[0] * 8] + pinyin_tokens + [[0] * 8]).view(-1)
            # input_ids, pinyin_ids = xnli_process.tokenizer.tokenize_sentence(senten[0],senten[1])
            length = input_ids.shape[0]
            input_ids = input_ids.view(1, length)
            pinyin_ids = pinyin_ids.view(1, length, 8)
            if torch.cuda.is_available():  # GPU available
                pinyin_ids = pinyin_ids.to(torch.device('cuda'))
                input_ids = input_ids.to(torch.device('cuda'))
            y_output = self.model(input_ids, pinyin_ids)
            y_output = torch.cuda.FloatTensor(y_output[0])
            s = torch.exp(y_output)
            y_hat=s / torch.sum(s, dim=1, keepdim=True)
            output=y_hat.cpu().detach().numpy().tolist()[0]
            ret.append(np.array(output))
        
        return np.array(ret)



def dataset_mapping(x):
    return {
        "x": x["text_a"],
        "y": x["label"],
    }

import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--save_path", required=True, type=str, help="save path of results")
    parser.add_argument("--attacker", default="pwws", type=str, help="attack algorithms, pwws or textbugger or random")
    parser.add_argument("--model_path", required=True, type=str, help="path of model")
    parser.add_argument("--data_path", required=True, type=str, help="path of datasets")
    parser.add_argument("--data_num", default=1000, type=int, help="path of datasets")    
    parser.add_argument("--mode", default="attack", type=str, help="attack or augmentation or representation")
    
    return parser
    
    
def main():
    parser = get_parser()
    args = parser.parse_args()

    print("Loading chinese processor and substitute")
    chinese_processor = OpenAttack.text_processors.ChineseTextProcessor()
    chinese_substitute = ChineseGlyphPronSubstitute()

    print("New Attacker")
    if args.attacker=="pwws":
        attacker = OpenAttack.attackers.PWWSAttacker(processor=chinese_processor, substitute=chinese_substitute,path=args.save_path,  threshold=40,mode=args.mode)
    elif args.attacker=="textbugger":
        attacker = OpenAttack.attackers.TextBuggerAttacker(processor=chinese_processor, substitute=chinese_substitute,path=args.save_path,  threshold=40,mode=args.mode)
    else:
        attacker = OpenAttack.attackers.RandomAttacker(processor=chinese_processor, substitute=chinese_substitute,path=args.save_path,  threshold=40,mode=args.mode)

    print("Building model")
    path='res/ckpts/checkpoint/ChnSetiCorp_model_adv_chinesebert.ckpt'
    with open("res/ckpts/checkpoint/args_chinesebert.json",'r') as load_f:
        dict = json.load(load_f)
        print(dict)
    model_args = argparse.Namespace(**dict)
    model = XNLITask(model_args)
    checkpoint = torch.load(args.model_path)#, map_location=lambda storage, loc: storage.cuda(0))
    model.load_state_dict(checkpoint['state_dict'])
    clsf = ChineseBert_model(model,model_args)
    split_num="train[:"+str(args.data_num)+"]"

    print("Loading dataset")
    dataset = datasets.load_dataset('csv', data_files=args.data_path,delimiter='\t',split=split_num).map(function=dataset_mapping)
    print(dataset)
    correct_samples = [
        inst for inst in dataset if (len(inst["x"])<512 and clsf.get_pred( [inst["x"]] )[0] == inst["y"])
    ]
    #print(correct_samples)
    print(len(correct_samples))

    print("Start attack")
    options = {
        "success_rate": True,
        "fluency": False,   # optional; see README "Optional evaluation metrics"
        "mistake": False,   # optional; see README "Optional evaluation metrics"
        "semantic": False,  # optional; see README "Optional evaluation metrics"
        "levenstein": True,
        "word_distance": True,
        "modification_rate": True,
        "running_time": True,
    }
    attack_eval = OpenAttack.attack_evals.ChineseAttackEval(attacker, clsf, **options, num_process=1)
    attack_eval.eval(correct_samples, visualize=True, progress_bar=True)

if __name__ == "__main__":
    main()
