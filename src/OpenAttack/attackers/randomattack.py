from ..attacker import Attacker
from ..text_processors import DefaultTextProcessor
from ..data_manager import DataManager
from ..substitutes import CounterFittedSubstitute
from ..substitutes.wordnet import WordNetSubstitute
from ..exceptions import WordNotInDictionaryException
import random
import numpy as np
import re
import zhon.hanzi
from ..utils import check_parameters
import Levenshtein

def write_advfile(filepath,data,dis):
    if filepath is None:
        return
    with open(filepath,'a+',encoding='utf8') as f:
        for it in data:
            f.write(str(dis)+' '+str(it[0])+' '+it[1]+'\n')

def write_file(filepath,sentence,dis):
    # filepath is None unless the caller opted in: this writes a plaintext copy
    # of the original/adversarial texts to disk.
    if filepath is None:
        return
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
    "processor": DefaultTextProcessor(),
    "threshold": 40,
    # Accepted for interface parity with the other attackers; nothing is printed.
    "verbose": False,
    # Scratch file the attacker appends each successful text to. None (the
    # default) disables it: enabling stores original/adversarial text in plaintext.
    "path": None,
    # Set by the drivers to record which stage produced the run; unused here.
    "mode": None
}


class RandomAttacker(Attacker):
    def __init__(self, **kwargs):
        """
        :param bool blackbox: Classifier Capacity. True-probability; False-grad. **Default:** True.

        :Classifier Capacity: Blind or Probability

        TEXTBUGGER: Generating Adversarial Text Against Real-world Applications. Jinfeng Li, Shouling Ji, Tianyu Du, Bo Li, Ting Wang. NDSS 2019.
        `[pdf] <https://arxiv.org/pdf/1812.05271.pdf>`__
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        # self.textprocessor = self.config["textprocessor"]
        # self.counterfit = CounterFittedSubstitute()
        self.glove_vectors = None
        # self.config = DEFAULT_CONFIG.copy()
        # self.config.update(kwargs)
        if self.config["substitute"] is None:
            self.config["substitute"] = WordNetSubstitute()
        check_parameters(DEFAULT_CONFIG.keys(), self.config)

        self.processor = self.config["processor"]
        self.substitute = self.config["substitute"]
        self.filepath=self.config.get("path")

    def __call__(self, clsf, x_orig, target=None):
        """
        * **clsf** : **Classifier** .
        * **x_orig** : Input sentence.
        """
        result, _trace = self.attack_with_trace(clsf, x_orig, target)
        return result

    def attack_with_trace(self, clsf, x_orig, target=None):
        """Same attack as __call__, plus the ordered intermediate texts (one per
        substitution), the last being the successful adversarial text.

        Word order is randomised, so results depend on the RNG state; callers
        seed per example for reproducibility.

        :return: ``(result, trace)`` — ``result`` is ``None`` or
            ``(adversarial_text, pred)``, identical to __call__.
        """
        trace = []
        #y_orig = clsf.get_pred([x_orig])[0]
        y_orig=clsf.get_prob([x_orig]).argmax(axis=1)
        # x = self.tokenize(x_orig)
        x = self.processor.get_tokens(x_orig)
        poss =  list(map(lambda xx: xx[1], x)) 
        x =  list(map(lambda xx: xx[0], x)) 
        random_pos=[i for i in range(len(x))]
        random.shuffle(random_pos)
        samples=[]
        for i in random_pos:
            word=x[i]
            candidate= list(map(lambda x:x[0], self.substitute(word, 0, threshold = self.config["threshold"])))
            if not candidate:
                continue
            x[i]=random.choice(candidate)
            #print(x)
            x_sentence = self.processor.detokenizer(x)
            prediction=clsf.get_prob([x_sentence]).argmax(axis=1)
            samples.append([y_orig[0],x_sentence])
            trace.append(x_sentence)
            #print(samples)
            if target is None:
                if prediction != y_orig:
                    #sentence="".join(ret_sent)
                    dis=Levenshtein.distance(x_orig,x_sentence)
                    dis=dis/max(len(x_sentence),1)
                    write_file(self.filepath,x_sentence,dis)
                    #write_advfile(self.filepath,samples,dis)
                    return (self.processor.detokenizer(x), prediction), trace
            else:
                if int(prediction) == int(target):
                    dis=Levenshtein.distance(x_orig,x_sentence)
                    dis=dis/max(len(x_sentence),1)
                    write_file(self.filepath,x_sentence,dis)
                    #write_advfile(self.filepath,samples,dis)
                    return (self.processor.detokenizer(x), prediction), trace

        return None, trace