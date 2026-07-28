import numpy as np
from ..text_processors import DefaultTextProcessor
from ..substitutes import WordNetSubstitute
from ..utils import check_parameters
from ..attacker import Attacker
from ..exceptions import WordNotInDictionaryException
import Levenshtein

DEFAULT_CONFIG = {
    "threshold": 0.5,
    "processor": DefaultTextProcessor(),
    "substitute": None,
    "token_unk": "<UNK>",
    # Print original/adversarial texts and probabilities while attacking. Off by
    # default: these lines would put dataset content into terminal and job logs.
    "verbose": False,
    # Scratch file the attacker appends each successful text to. None (the
    # default) disables it: enabling stores original/adversarial text in plaintext.
    "path": None,
    # Set by the drivers to record which stage produced the run; unused here.
    "mode": None
}

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

class PWWSAttacker(Attacker):
    def __init__(self,**kwargs):
        """
        :param float threshold: Threshold used in substitute module. **Default:** 0.5
        :param WordSubstitute substitute: Substitute method used in this attacker. **Default:** :any:`WordNetSubstitute`
        :param TextProcessor processor: Text processor used in this attacker. **Default:** :any:`DefaultTextProcessor`
        :param str token_unk: A token which means "unknown token" in Classifier's vocabulary.

        :Classifier Capacity: Probability

        Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency. Shuhuai Ren, Yihe Deng, Kun He, Wanxiang Che. ACL 2019.
        `[pdf] <https://www.aclweb.org/anthology/P19-1103.pdf>`__
        `[code] <https://github.com/JHL-HUST/PWWS/>`__
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        if self.config["substitute"] is None:
            self.config["substitute"] = WordNetSubstitute()
        check_parameters(DEFAULT_CONFIG.keys(), self.config)

        self.processor = self.config["processor"]
        self.substitute = self.config["substitute"]
        self.filepath=self.config.get("path")
        
    def __call__(self, clsf, x_orig, target=None):
        # Unchanged public contract: returns None or (adversarial_text, pred).
        # The intermediate trace is computed but discarded here; callers that
        # want it (e.g. build_augmented_data.py) use attack_with_trace instead.
        result, _trace = self.attack_with_trace(clsf, x_orig, target)
        return result

    def attack_with_trace(self, clsf, x_orig, target=None):
        """Same attack as __call__, but also returns the ordered list of
        intermediate perturbed texts (each word-substitution step), with the
        final successful adversarial text as the last element.

        :return: ``(result, trace)`` where ``result`` is ``None`` or
            ``(adversarial_text, pred)`` (identical to __call__), and ``trace``
            is a list[str] of every intermediate text in generation order
            (empty if the attack failed / produced no steps).
        """
        x_orig = x_orig.lower()
        sentence_ori=x_orig
        trace = []
        #print(x_orig)
        if target is None:
            targeted = False
            target = clsf.get_pred([x_orig])[0]  # calc x_orig's prediction
            conf=clsf.get_prob([x_orig])
            if self.config["verbose"]:
                print(target,conf,x_orig)
        else:
            targeted = True

        #x_orig = list(map(lambda x: x[0], self.processor.get_tokens(x_orig)))   # tokenize
        x_orig = self.processor.get_tokens(x_orig)
        poss =  list(map(lambda x: x[1], x_orig))
        x_orig =  list(map(lambda x: x[0], x_orig))

        S = self.get_saliency(clsf, x_orig, target, targeted) # (len(sent), )
        #print("111")
        S_softmax = np.exp(S - S.max())
        S_softmax = S_softmax / S_softmax.sum()

        w_star = [ self.get_wstar(clsf, x_orig, i, poss[i], target, targeted) for i in range(len(x_orig)) ]  # (len(sent), )
        H = [ (idx, w_star[idx][0], S_softmax[idx] * w_star[idx][1]) for idx in range(len(x_orig)) ]
        #print("222")
        H = sorted(H, key=lambda x:-x[2])
        ret_sent = x_orig.copy()
        samples=[]
        #self.filepath='adv_text/samples_chinesebert.txt'
        for i in range(len(H)):
            idx, wd, _ = H[i]
            ret_sent[idx] = wd
            #print(ret_sent)
            sentence="".join(ret_sent)
            samples.append([target,sentence])

            # pred = clsf.get_pred([self.config["processor"].detokenizer(ret_sent)])[0]
            x_adv=self.config["processor"].detokenizer(ret_sent)
            # each perturbation step is an intermediate text (Algorithm 1, tmp)
            trace.append(x_adv)
            # pred=clsf.get_pred([x_adv])[0]
            # conf=clsf.get_pred([x_adv])[0]
            prob=clsf.get_prob([x_adv])

            pred=prob.argmax(axis=1)

            #print("pred",pred)
            if targeted:
                if pred == target:
                    sentence="".join(ret_sent)
                    dis=Levenshtein.distance(sentence,sentence_ori)
                    dis=dis/max(len(sentence),1)
                    write_file(self.filepath,sentence,dis)
                    #write_advfile(self.filepath,samples,dis)
                    return (self.config["processor"].detokenizer(ret_sent), pred), trace
            else:
                if pred != target:
                    #write_file(self.filepath,samples)
                    # x_adv=self.config["processor"].detokenizer(ret_sent)
                    # conf=clsf.get_pred([x_adv])
                    # print(target,pred,conf,x_adv)
                    if self.config["verbose"]:
                        print("conf",pred,prob,x_adv)
                    sentence="".join(ret_sent)
                    dis=Levenshtein.distance(sentence,sentence_ori)
                    dis=dis/max(len(sentence),1)
                    #write_advfile(self.filepath,samples,dis)
                    write_file(self.filepath,sentence,dis)
                    return (self.config["processor"].detokenizer(ret_sent), pred), trace
        return None, trace


    
    def get_saliency(self, clsf, sent, target, targeted):
        x_hat_raw = []
        for i in range(len(sent)):
            left = sent[:i]
            right = sent[i + 1:]
            x_i_hat = left + [self.config["token_unk"]] + right
            x_hat_raw.append(self.config["processor"].detokenizer(x_i_hat))
        x_hat_raw.append(self.config["processor"].detokenizer(sent))
        #print("fff",len(x_hat_raw))
        # res = clsf.get_prob(x_hat_raw)[:, target]
        res=[]
        for i in range(len(x_hat_raw)):
            c=x_hat_raw[i]
            res.append(clsf.get_prob([c])[0])
        res=np.array(res)[:,target]
        if not targeted:
            res = res[-1] - res[:-1]
        else:
            res = res[:-1] - res[-1]
        return res

    def get_wstar(self, clsf, sent, idx, pos, target, targeted):
        word = sent[idx]
        try:
            rep_words = list(map(lambda x:x[0], self.substitute(word, pos, threshold = self.config["threshold"])))
        except WordNotInDictionaryException:
            rep_words = []
        rep_words = list(filter(lambda x: x != word, rep_words))
        if len(rep_words) == 0:
            return ( word, 0 )
        sents = []
        for rw in rep_words:
            new_sent = sent[:idx] + [rw] + sent[idx + 1:]
            sents.append(self.config["processor"].detokenizer(new_sent))
        sents.append(self.config["processor"].detokenizer(sent))
        #res = clsf.get_prob(sents)[:, target]
        res=[]
        for c in sents:
            #c=sents[i]
            res.append(clsf.get_prob([c])[0])
        res=np.array(res)[:,target]


        prob_orig = res[-1]
        res = res[:-1]
        if targeted:
            return (rep_words[ res.argmax() ],  res.max() - prob_orig )
        else:
            return (rep_words[ res.argmin() ],  prob_orig - res.min() )
    








