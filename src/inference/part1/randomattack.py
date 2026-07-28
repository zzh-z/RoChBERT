from ..attacker import Attacker
from ..text_processors import DefaultTextProcessor
from ..data_manager import DataManager
from ..substitutes import CounterFittedSubstitute
from ..exceptions import WordNotInDictionaryException
import random
import numpy as np
import re
import zhon.hanzi
from ..utils import check_parameters
import Levenshtein

def write_advfile(filepath,data,dis):
    with open(filepath,'a+',encoding='utf8') as f:
        for it in data:
            f.write(str(dis)+' '+str(it[0])+' '+it[1]+'\n')

def write_file(filepath,sentence,dis):
    with open(filepath,'a+',encoding='utf8') as f:
        f.write(str(dis)+' '+sentence+'\n')

# def sent_tokenize(x):
#     sents_temp = re.split('(：|:|,|，|。|！|\!|\.|？|\?)', x)
#     sents = []
#     for i in range(len(sents_temp)//2):
#         sent = sents_temp[2*i] + sents_temp[2*i+1]
#         sents.append(sent)
#     return sents

def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")

DEFAULT_CONFIG = {
    "blackbox": True,
    "substitute": None,
    "processor": DefaultTextProcessor()
}


class RandomAttacker(Attacker):
    def __init__(self, **kwargs):
        """
        :param bool blackbox: Classifier Capacity. True-probability; False-grad. **Default:** True.

        :Data Requirements: :py:data:`.TProcess.NLTKSentTokenizer`
        :Classifier Capacity: Blind or Probability

        TEXTBUGGER: Generating Adversarial Text Against Real-world Applications. Jinfeng Li, Shouling Ji, Tianyu Du, Bo Li, Ting Wang. NDSS 2019.
        `[pdf] <https://arxiv.org/pdf/1812.05271.pdf>`__
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        self.nlp = DataManager.load("TProcess.NLTKSentTokenizer")
        # self.textprocessor = self.config["textprocessor"]
        # self.counterfit = CounterFittedSubstitute()
        self.glove_vectors = None
        # self.config = DEFAULT_CONFIG.copy()
        # self.config.update(kwargs)
        if self.config["substitute"] is None:
            self.config["substitute"] = WordNetSubstitute()
        check_parameters(self.config.keys(), DEFAULT_CONFIG)

        self.processor = self.config["processor"]
        self.substitute = self.config["substitute"]
        self.filepath=self.config["path"]
        self.mode=self.config["mode"]

    def __call__(self, clsf, x_orig, target=None):
        """
        * **clsf** : **Classifier** .
        * **x_orig** : Input sentence.
        """
        #y_orig = clsf.get_pred([x_orig])[0]
        y_orig=clsf.get_prob([x_orig]).argmax(axis=1)
        # x = self.tokenize(x_orig)
        x = self.processor.get_tokens(x_orig)
        poss =  list(map(lambda xx: xx[1], x)) 
        x =  list(map(lambda xx: xx[0], x)) 
        random_pos=[i for i in range(len(x))]
        random.shuffle(random_pos)
        samples=[]

        sep_word_idx=len(x)
        sep_char_idx=len(x_orig)
        if "sep" in x_orig:
            senten=x_orig.split("[sep]")
            sep_word_idx=x.index("sep")+1
            sep_char_idx=x_orig.find("[sep]")
        sep_len=len(x_orig)-sep_char_idx-5
        
        for i in random_pos:
            if i<=sep_word_idx:
                continue
            word=x[i]
            candidate= list(map(lambda x:x[0], self.substitute(word, 0, threshold = 1)))
            x[i]=candidate[0]
            #print(x)
            x_sentence = self.processor.detokenizer(x)
            prediction=clsf.get_prob([x_sentence]).argmax(axis=1)
            samples.append([y_orig[0],x_sentence])
            #print(samples)
            if target is None:
                if prediction != y_orig:
                    #sentence="".join(ret_sent)
                    dis=Levenshtein.distance(x_orig,x_sentence)
                    dis=dis/sep_len
                    # write_file(self.filepath,x_sentence,dis)
                    #write_advfile(self.filepath,samples,dis)
                    if self.mode=="attack":
                            write_file(self.filepath,x_sentence,dis)
                    elif self.mode=="augmentation":
                        write_advfile(self.filepath,samples,dis)
                    else:
                        write_file(self.filepath,x_orig,y_orig[0])
                        write_file(self.filepath,x_prime_sentence,dis)

                    return self.processor.detokenizer(x), prediction
            else:
                if int(prediction) is int(target):
                    dis=Levenshtein.distance(x_orig,x_sentence)
                    dis=dis/sep_len
                    # write_file(self.filepath,x_sentence,dis)
                    # #write_advfile(self.filepath,samples,dis)
                    if self.mode=="attack":
                            write_file(self.filepath,x_sentence,dis)
                    elif self.mode=="augmentation":
                        write_advfile(self.filepath,samples,dis)
                    else:
                        write_file(self.filepath,x_orig,y_orig[0])
                        write_file(self.filepath,x_prime_sentence,dis)

                    return self.processor.detokenizer(x), prediction
        
        if self.mode=="representation":
            write_file(self.filepath,x_orig,y_orig[0])
            write_file(self.filepath,x_orig,1)

        return None