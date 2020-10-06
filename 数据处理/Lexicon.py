# -*- coding: utf-8 -*-
__author__ = "Yue Yuanhao"
__copyright__ = "Copyright (C) 2020 Yue Yuanhao"
__license__ = "Public Domain"
__version__ = "1.5"

import numpy as np
import os
from itertools import chain
from pypinyin import pinyin, Style


def to_pinyin(s):
    '''
    :param s: 字符串或列表
    :type s: str or list
    :return: 拼音字符串
    '''
    return ''.join(chain.from_iterable(pinyin(s, style=Style.TONE3)))

class Lexicon:
    def __init__(self, input_path='./dict/中文词典2.txt', saved_path='./save/lexicon_check.npy', save_path='./save'):
        """
        :param input_path: 词典的路径
        """
        self.input_path = input_path
        self.saved_path = saved_path
        self.save_path = save_path
        if os.path.isfile(saved_path):
            self.word_dict = np.load(saved_path, allow_pickle=True).item()
        else:
            self.word_dict = self.build_lexicon(self.input_path)

    def build_lexicon(self, input_path):
        """
        :param input_path: 词典的路径
        :return: 返回word_dict
        """
        self.sort(input_path)
        file = open(self.save_path+'/sorted_dictionary.txt', 'r')
        self.word_dict = dict()
        try:
            text = file.read().split()
            for i, word in enumerate(text):
                if i == len(text) - 1:
                    self.word_dict[word] = 1
                elif word in text[i + 1]:
                    self.word_dict[word] = 2
                elif not word in text[i + 1]:
                    self.word_dict[word] = 1

                if len(word) > 2:
                    for j in range(len(word)):
                        if not word[:j] in self.word_dict.keys():
                            self.word_dict[word[:j]] = 3
                    self.word_dict[word] = 1
        finally:
            file.close()
        self.save_dictionary(self.save_path)

    def load_dictionary(self, saved_path):
        """
        :param saved_dictionary: 已保存词典的路径
        :return: 是否加载成功
        加载已保存词典
        """
        self.word_dict = np.load(saved_path, allow_pickle=True).item()

    def save_dictionary(self, save_path):
        """
        :param save_path: 保存词典的路径
        :return: 是否保存成功
        保存词典至指定路径
        """
        np.save(save_path + '/lexicon_check.npy', self.word_dict)

    def search(self, query):
        """
        :param query: str
        :return: 输入一个潜在的词语(str), 返回值类型为int
        0 无此词汇
        1 有且确定此词汇
        2 有且有更大的词汇
        3 无此词汇，有更大的词汇
        """
        if self.word_dict.get(query):
            return self.word_dict[query]
        else:
            return 0


    def match(self,sentence):
        """
        :param sentence: str, 包含一个句子
        :param lex: Lexicon
        :return: 返回该句子中出现的单词的列表
        该函数将调用Lexicon.search(), 保留所有在输入序列中确实存在的词语，并将这些词语用来构建Data.words
        """
        word_list = []
        pair_list = []
        if len(sentence) < 2:
            raise Exception("句子太短")
        length = len(sentence)
        for loc in range(length - 1):
            for step in range(1, length - loc):
                cur_word = sentence[loc:loc + step+1]
                code = self.search(cur_word)
                if code == 0:
                    break
                if code == 1:
                    word_list.append(cur_word)
                    pair_list.append((loc, loc + step))
                    break
                if code == 2:
                    word_list.append(cur_word)
                    pair_list.append((loc, loc + step))
                    continue
                if code == 3:
                    continue
        return word_list, pair_list

    def sort(self,input_path):
        file = open(input_path, 'r')
        text = file.read().split()
        file.close()
        text.sort()
        text=sorted(text, key=to_pinyin)
        outfile=open(self.save_path+'/sorted_dictionary.txt', 'w')
        for l in text:
            outfile.write(l)
            outfile.write('\n')

if __name__=='__main__':
    lex = Lexicon(input_path='./dict/中文词典2.txt', saved_path='./save/lexicon_check.npy', save_path='./save')
    print('input: 在长江中下游平原')
    out = lex.match('在长江中下游平原')
    print('output: ', end='')
    print(out)
    print(lex.word_dict)

'''
返回 0 1 2 3
0 无此词汇
1 有且确定此词汇
2 有且有更大的词汇
3 无此词汇，有更大的词汇
'''

'''
输入的Lexicon必须是有序的！
'''

'''
词典不全，没有中国这个词
search函数load的问题
'''
print('')