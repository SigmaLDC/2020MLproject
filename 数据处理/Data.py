# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:06:33 2020

@author: Ryan He
"""
import Lexicon
import time


class Data:
    def __init__(self, file_path, words_dict):
        with open(file_path, 'r', encoding='utf-8') as file:
            input_file = file.read()
        input_file = input_file.split('\n')
        self.properties = dict()
        self.chars_dict = self.build_chars_dict(input_file)
        # self.words_dict = word_dict
        self.labels_dict = self.build_labels_dict(input_file)
        self.data = self.build_structured_data(input_file, words_dict)
        # self.chars_list = self.build_chars_list(input_file)
        # self.words_list = self.build_words_list(input_file)
        # self.labels_list = self.build_labels_list(input_file)

    def build_chars_dict(self, input_file):
        """
        :param input_file: 输入文件的列表形式
        :return: 字典, 用于初始化chars
        建立chars字典, 字典的key是字符, value是对应的index, index从1开始
        """
        chars_dict = {}
        index = 1

        for item in input_file:
            if item != '':
                char = item[0]
                if chars_dict.get(char, 0) == 0:
                    chars_dict[char] = index
                    index += 1

        return chars_dict

    # def build_words_dict(self, input_file, word_dict):
    #    """
    #    :param input_file: 输入文件的字符串形式
    #    :param word_dict: Lexicon类的实例
    #    :return: 字典, 用于初始化words
    #    建立words字典, 字典的key是(beg_index, end_index), 其中beg_index是词的首字对应的index, end_index是词的尾字对应的index
    #    字典的value是词的index(即该词是第几个词), index从1开始
    #    """
    #    pass

    def build_labels_dict(self, input_file):
        """
        :param input_file: 输入文件的列表形式
        :return: 字典, 用于初始化labels
        建立labels字典, 字典的key是标签, value是其对应的index, index从1开始
        """
        labels_dict = {}
        index = 1

        for item in input_file:
            # if item != '':
            if len(item) >= 3:
                label = item[2:]
                if labels_dict.get(label, 0) == 0:
                    labels_dict[label] = index
                    index += 1

        return labels_dict

    def build_structured_data(self, input_file, words_dict):
        """
        :param words_dict: 词典 :param input_file: 输入文件的列表形式 :return: 结构化的数据, 数据结构为:
        [sentence[[chars_list], [chars_index(自增)], [words_list(pair)], [words_index(lexicon)], [labels_list]]]
        """
        sentences = []
        max_length = 0
        sentence = []
        length = 0

        for item in input_file:
            if item != '':
                sentence.append(item)
                length += 1
            elif length > 0:
                sentences.append(sentence)
                if length > max_length:
                    max_length = length
                sentence = []
                length = 0
            else:
                continue

        self.properties['max_sentence_length'] = max_length
        self.properties['sentence_number'] = len(sentences)

        data = []
        max_number = 0
        for sentence in sentences:
            temp = []
            chars_list = self.build_chars_list(sentence)
            words_list, words_list_lexicon, words_number = self.build_words_list(sentence, words_dict)
            labels_list = self.build_labels_list(sentence)
            temp.append(chars_list)
            temp.append(list(range(len(chars_list))))
            temp.append(words_list)
            temp.append(words_list_lexicon)
            temp.append(labels_list)
            data.append(temp)
            if words_number > max_number:
                max_number = words_number
        self.properties['max_words_number'] = max_number
        return data

    def build_chars_list(self, input_sentence):
        """
        :param input_sentence: 句子的列表形式
        :return: 返回包含所有字的列表, 列表元素为该字符在chars_dict中对应的index
        """
        chars_list = []

        for item in input_sentence:
            index = self.chars_dict[item[0]]
            chars_list.append(index)

        return chars_list

    def build_words_list(self, input_sentence, words_dict):
        """
        :param words_dict: 词典
        :param input_sentence: 句子的列表形式
        :return: 返回包含所有在输入数据中出现的单词的列表, 列表的元素为(beg_index, end_index)
                 其中beg_index是单词首字在chars_list中的位置, end_index是单词尾字在chars_list中的位置, 位置从0开始
                 返回单词在lexicon中的index组成的列表
                 返回句子包含的单词数
        """
        words_list = []
        sentence_str = ''
        for item in input_sentence:
            sentence_str += item[0]
        # raw_words, words_index = words_dict.match(sentence_str)
        raw_words, words_list = words_dict.match(sentence_str)
        # raw_words = list(set(raw_words))  # 消重

        # for word in raw_words:
        #     beg_index = self.chars_dict[word[0]]
        #     end_index = self.chars_dict[word[-1]]
        #     words_list.append((beg_index, end_index))
        words_index = []
        key_list = list(lex.word_dict.keys())
        for word in raw_words:
            index = key_list.index(word)
            words_index.append(index)

        return words_list, words_index, len(raw_words)

    def build_labels_list(self, input_sentence):
        """
        :param input_sentence: 句子的列表形式
        :return: 返回包含所有char对应label的列表, 列表元素为该label在labels_dict中对应的index
        """
        labels_list = []

        for item in input_sentence:
            try:
                index = self.labels_dict[item[2:]]
            except KeyError:
                index = self.labels_dict['O']
            labels_list.append(index)

        return labels_list

    def show_data_info(self):
        """
        :return: None
        显示Data对象的信息, 包括句子最大长度, 单个句子中words的最大长度, 输入文件中句子数目
        """
        print('Data信息:')
        print('句子最大长度:', self.properties['max_sentence_length'])
        print('单个句子中最大单词个数:', self.properties['max_words_number'])
        print('句子数目:', self.properties['sentence_number'])


if __name__ == '__main__':
    tic = time.time()
    print('测试Data类...')
    lex = Lexicon.Lexicon()
    print(1)
    path = r'NERData\MSRA\msra_train_bio.txt'
    data = Data(path, lex)
    data.show_data_info()
    print(data.chars_dict, data.labels_dict, sep='\n')
    for i in data.data[0]:
        print(i)
    toc = time.time()
    print('运行时间:', toc - tic)
