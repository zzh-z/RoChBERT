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


class TextBuggerAttacker(Attacker):
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
        #sentences_of_doc = self.get_sentences(x)
        #sentences_of_doc=re.findall(zhon.hanzi.sentence, x_orig)
        # sentences_of_doc=cut_sent(x_orig)
        # print(sentences_of_doc)
        # if len(sentences_of_doc)==0:
        #     sentences_of_doc=[x_orig]
        sentences_of_doc=[x_orig]
        ranked_sentences = self.rank_sentences(sentences_of_doc, clsf, y_orig)
        x_prime = x.copy()
        samples=[]

        sep_word_idx=len(x_prime)
        sep_char_idx=len(x_orig)
        if "sep" in x_orig:
            senten=x_orig.split("[sep]")
            sep_word_idx=x_prime.index("sep")-1
            sep_char_idx=x_orig.find("[sep]")
        sep_len=sep_char_idx
        

        for sentence_index in ranked_sentences:
            # if self.config["blackbox"] is True:
            ranked_words = self.get_word_importances(sentences_of_doc,sentence_index, clsf, y_orig)
            # else:
            #     ranked_words = self.get_w_word_importances(sentences_of_doc[sentence_index], clsf, y_orig)
            for word, loss in ranked_words:
                bug = self.selectBug(word, x_prime, clsf)
                x_prime = self.replaceWithBug(x_prime, word, bug)
                x_prime_sentence = self.processor.detokenizer(x_prime)
                #prediction = clsf.get_pred([x_prime_sentence])[0]
                prediction=clsf.get_prob([x_prime_sentence]).argmax(axis=1)
                samples.append([y_orig[0],x_prime_sentence])
                #print(clsf.get_prob([x_orig]),x_orig)
                #print(clsf.get_prob([x_prime_sentence]),x_prime_sentence)

                if target is None:
                    if prediction != y_orig:
                        #sentence="".join(ret_sent)
                        dis=Levenshtein.distance(x_orig,x_prime_sentence)
                        dis=dis/sep_len
                        
                        if self.mode=="attack":
                            write_file(self.filepath,x_prime_sentence,dis)
                        elif self.mode=="augmentation":
                            write_advfile(self.filepath,samples,dis)
                        else:
                            write_file(self.filepath,x_orig,y_orig[0])
                            write_file(self.filepath,x_prime_sentence,dis)
                        #write_advfile(self.filepath,samples,dis)

                        return self.processor.detokenizer(x_prime), prediction
                else:
                    if int(prediction) is int(target):
                        dis=Levenshtein.distance(x_orig,x_prime_sentence)
                        dis=dis/sep_len
                        # write_file(self.filepath,x_prime_sentence,dis)
                        # #write_advfile(self.filepath,samples,dis)
                        if self.mode=="attack":
                            write_file(self.filepath,x_prime_sentence,dis)
                        elif self.mode=="augmentation":
                            write_advfile(self.filepath,samples,dis)
                        else:
                            write_file(self.filepath,x_orig,y_orig[0])
                            write_file(self.filepath,x_prime_sentence,dis)

                        return self.processor.detokenizer(x_prime), prediction
        if self.mode=="representation":
            write_file(self.filepath,x_orig,y_orig[0])
            write_file(self.filepath,x_orig,1)
        return None

    def get_sentences(self, x):
        return [self.textprocessor.detokenizer(x)]

    def rank_sentences(self, sentences, clsf, target_all):
        map_sentence_to_loss = {}  # 与原文不同
        for i in range(len(sentences)):
            #y_orig = clsf.get_pred([sentences[i]])[0]
            y_orig=clsf.get_prob([sentences[i]]).argmax(axis=1)
            if y_orig != target_all:
                continue
            tempoutput = clsf.get_prob([sentences[i]])[0]
            ret = max(tempoutput[target_all], 1e-3)
            map_sentence_to_loss[i] = ret
        sentences_sorted_by_loss = {k: v for k, v in sorted(map_sentence_to_loss.items(), key=lambda item: item[1], reverse=True)}
        return sentences_sorted_by_loss

    def get_word_importances(self, doc,sent_idx, clsf, y_orig):
        sentence_tokens = self.tokenize(doc[sent_idx])
        word_losses = {}
        for curr_token in sentence_tokens:
            sentence_tokens_without = [token for token in sentence_tokens if token != curr_token]
            sentence_without = self.processor.detokenizer(sentence_tokens_without)
            
            doc_without=""
            for idx in range(len(doc)):
                if idx!=sent_idx:
                    doc_without+=doc[idx]
                else:
                    doc_without+=sentence_without
            #print(doc_without)
            tempoutput = clsf.get_prob([doc_without])[0]
            ret = max(tempoutput[y_orig], 1e-3)
            word_losses[curr_token] = -ret #越大越重要
        word_losses = [(k, v) for k, v in sorted(word_losses.items(), key=lambda item: item[1], reverse=True)]
    #   print(word_losses)
        return word_losses

    def get_w_word_importances(self, sentence, clsf, y_orig):  # white
        from collections import OrderedDict

        sentence_tokens = self.tokenize(sentence)
        prob, grad = clsf.get_grad([sentence_tokens], [y_orig])
        grad = grad[0]
        dist = []
        for i in range(len(grad)):
            dist.append(0.0)
            for j in range(len(grad[i])):
                dist[i] += grad[i][j] * grad[i][j]
            dist[i] = np.sqrt(dist[i])
        word_losses = OrderedDict()
        for i, curr_token in enumerate(sentence_tokens):
            if i < len(dist):
                word_losses[curr_token] = dist[i]
            else:
                word_losses[curr_token] = 0
        word_losses = {(k, v) for k, v in sorted(word_losses.items(), key=lambda item: -item[1], reverse=True)}
        return word_losses

    def getSemanticSimilarity(x, x_prime, epsilon):
        return epsilon + 1

    def selectBug(self, original_word, x_prime, clsf):
        #bugs = self.generateBugs(original_word, self.glove_vectors)
        try:
            bugs = list(map(lambda x:x[0], self.substitute(original_word, 0, threshold = self.config["threshold"])))
        except WordNotInDictionaryException:
            bugs = []
            
        x_prime_sentence = self.processor.detokenizer(x_prime)
        # y_orig=clsf.get_prob([x_prime_sentence]).argmax(axis=1)
        tempoutput = max(clsf.get_prob([x_prime_sentence])[0])
        max_score = -max(tempoutput, 1e-3)   
        best_bug = original_word
        bug_tracker = {}
        for b_k in bugs:
            candidate_k = self.getCandidate(original_word, b_k, x_prime)
            score_k = self.getScore(candidate_k, x_prime, clsf)
            if score_k > max_score:
                best_bug = b_k
                max_score = score_k
            bug_tracker[b_k] = score_k
        return best_bug

    def getCandidate(self, original_word, new_bug, x_prime):
        sep_word_idx=x_prime.index("sep")-1
        tokens = x_prime
        new_tokens = [new_bug if tokens[i] == original_word and i<sep_word_idx else tokens[i] for i in range(len(tokens))]
        return new_tokens

    def getScore(self, candidate, x_prime, clsf):
        x_prime_sentence = self.processor.detokenizer(x_prime)
        candidate_sentence = self.processor.detokenizer(candidate)
        #y_orig = clsf.get_pred([x_prime_sentence])[0]
        y_orig=clsf.get_prob([x_prime_sentence]).argmax(axis=1)
        tempoutput = clsf.get_prob([candidate_sentence])[0]
        ret = max(tempoutput[y_orig], 1e-3)
        return -ret

    def replaceWithBug(self, x_prime, x_i, bug):
        sep_word_idx=x_prime.index("sep")-1
        tokens = x_prime
        new_tokens = [bug if tokens[i] == x_i and i<sep_word_idx else tokens[i] for i in range(len(tokens))]
        return new_tokens

    def generateBugs(self, word, glove_vectors, sub_w_enabled=False, typo_enabled=False):
        bugs = {"insert": word, "delete": word, "swap": word, "sub_C": word, "sub_W": word}
        if len(word) <= 2:
            return bugs
        bugs["insert"] = self.bug_insert(word)
        bugs["delete"] = self.bug_delete(word)
        bugs["swap"] = self.bug_swap(word)
        bugs["sub_C"] = self.bug_sub_C(word)
        bugs["sub_W"] = self.bug_sub_W(word)
        return bugs

    def bug_sub_W(self, word):
        try:
            res = self.counterfit(word, None, threshold=0.5)
            if len(res) == 0:
                return word
            return res[0][0]
        except WordNotInDictionaryException:
            return word

    def bug_insert(self, word):
        if len(word) >= 6:
            return word
        res = word
        point = random.randint(1, len(word) - 1)
        res = res[0:point] + " " + res[point:]
        return res

    def bug_delete(self, word):
        res = word
        point = random.randint(1, len(word) - 2)
        res = res[0:point] + res[point + 1:]
        return res

    def bug_swap(self, word):
        if len(word) <= 4:
            return word
        res = word
        points = random.sample(range(1, len(word) - 1), 2)
        a = points[0]
        b = points[1]

        res = list(res)
        w = res[a]
        res[a] = res[b]
        res[b] = w
        res = ''.join(res)
        return res

    def bug_sub_C(self, word):
        res = word
        key_neighbors = self.get_key_neighbors()
        point = random.randint(0, len(word) - 1)

        if word[point] not in key_neighbors:
            return word
        choices = key_neighbors[word[point]]
        subbed_choice = choices[random.randint(0, len(choices) - 1)]
        res = list(res)
        res[point] = subbed_choice
        res = ''.join(res)

        return res

    def get_key_neighbors(self):
        # By keyboard proximity
        neighbors = {
            "q": "was", "w": "qeasd", "e": "wrsdf", "r": "etdfg", "t": "ryfgh", "y": "tughj", "u": "yihjk",
            "i": "uojkl", "o": "ipkl", "p": "ol",
            "a": "qwszx", "s": "qweadzx", "d": "wersfxc", "f": "ertdgcv", "g": "rtyfhvb", "h": "tyugjbn",
            "j": "yuihknm", "k": "uiojlm", "l": "opk",
            "z": "asx", "x": "sdzc", "c": "dfxv", "v": "fgcb", "b": "ghvn", "n": "hjbm", "m": "jkn"
        }
        # By visual proximity
        neighbors['i'] += '1'
        neighbors['l'] += '1'
        neighbors['z'] += '2'
        neighbors['e'] += '3'
        neighbors['a'] += '4'
        neighbors['s'] += '5'
        neighbors['g'] += '6'
        neighbors['b'] += '8'
        neighbors['g'] += '9'
        neighbors['q'] += '9'
        neighbors['o'] += '0'

        return neighbors

    def tokenize(self, sent):
        tokens = list(map(lambda x: x[0], self.processor.get_tokens(sent)))
        return tokens



