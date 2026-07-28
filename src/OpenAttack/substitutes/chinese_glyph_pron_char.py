import random
from .base import WordSubstitute
from ..data_manager import DataManager
from ..exceptions import WordNotInDictionaryException
import json
from pypinyin import lazy_pinyin
import collections
import zhconv

class ChineseGlyphPronSubstitute(WordSubstitute):
    """
    :Data Requirements: :py:data:`.AttackAssist.CiLin`

    An implementation of :py:class:`.WordSubstitute`.

    Chinese Sememe-based word substitute based CiLin.
    """

    def __init__(self):
        super().__init__()
        # #self.cilin_dict = DataManager.load("AttackAssist.CiLin")
        # with open('res/process_dict/glyph_sim_chars_confirmed.json','r') as glyph_file:
        #     self.glyph_dict=json.load(glyph_file)
            
        # with open('res/process_dict/pronunciation_dict_non_tone.json','r') as pron_file:
        #     self.pron_dict=json.load(pron_file)

        # with open('res/process_dict/char_freq.json','r') as freq_file:
        #     self.freq_dict=json.load(freq_file,object_pairs_hook=collections.OrderedDict)

        with open('res/process_dict/output_dict.json','r') as char_file:
            self.char_dict=json.load(char_file,object_pairs_hook=collections.OrderedDict)

    def __call__(self, word, pos_tag, threshold=40):
        """
        :param word: the raw word; pos_tag: part of speech of the word, threshold: return top k words.
        :return: The result is a list of tuples, *(substitute, 1)*.
        :rtype: list of tuple
        """
        #print(word)
        char_candidate=[]
        if len(word)>4:
            flag=True
        else:
            flag=False
        # for char in word:
        #     if char not in self.glyph_pron_dict:
        #         print(1111111)
        #         raise WordNotInDictionaryException
        #     sym_char=self.glyph_pron_dict[char]
        #     char_candidate.append(sym_char)
        for char in word:
            # if char in self.glyph_dict:
            #     glyph_can=self.glyph_dict[char]
            # else:
            #     glyph_can=[char]
            # try:
            #     pinyin=lazy_pinyin(char)[0]
            #     #print(pinyin)
            #     pron_can=self.pron_dict[pinyin]
            #     #pron_can.sort()
            # except:
            #     pron_can=[]
            # all_can=glyph_can+pron_can
            # #all_can.sort()
            # all_can="".join(list(set(all_can)))
            # #all_can=list(set(all_can))
            # all_can=zhconv.convert(all_can,'zh-hans')
            # freq_can=[]
            # for c in all_can:
            #     if c not in self.freq_dict or self.freq_dict[c]<200:
            #         continue
            #     freq_can.append(c)
            # freq_can.sort()
            # freq_can="".join(list(set(freq_can)))
            # #print(all_can)
            # #freq_can.sort()
            if char in self.char_dict:
                freq_can=self.char_dict[char]
            else:
                freq_can=[char]
            if flag:
                freq_can=freq_can[:10]
            char_candidate.append(freq_can)

        
        #char_candidate = ['123','78','few']
        # `threshold` bounds the number of candidates returned (at most 40, per
        # Appendix A.2), so it is applied while building rather than after: the
        # full Cartesian product reaches c^len(word) strings and exhausts memory
        # on long jieba tokens (中国社会科学出版社 gives 8e8 combinations).
        # Loop order is unchanged, so ret[:threshold] is identical either way.
        # threshold=None builds the full product.
        sym_words=['']
        for i in range(0, len(char_candidate)):
            tmp=[]
            for j in range(0,len(char_candidate[i])):
                for k in range(0,len(sym_words)):
                    tmp.append(sym_words[k]+char_candidate[i][j])
                    if threshold is not None and len(tmp) >= threshold:
                        break
                if threshold is not None and len(tmp) >= threshold:
                    break
            sym_words=tmp

        # if word not in self.cilin_dict:
        #     raise WordNotInDictionaryException()
        # sym_words = self.cilin_dict[word]

        ret = []
        #ret [('另一方面', 1), ('单方面', 1), ('一端', 1), ('一边', 1), ('一派', 1), ('单', 1), ('一头', 1), ('单向', 1), ('一面', 1)]
        for sym_word in sym_words:
            ret.append((sym_word, 1))
        if threshold is not None:
            ret = ret[:threshold]
        #print(word,len(ret))
        return ret
