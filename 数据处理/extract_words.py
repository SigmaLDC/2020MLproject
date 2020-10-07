"""从训练集中抽取单词"""
import os

if __name__ == '__main__':
    # path = input('输入要抽取单词的文件路径: ')
    path = "NERData\\Weibo\\simple_weiboNER_2nd_conll.train"
    file = open(path, 'r', encoding='utf-8')
    file_str = file.read()
    file_list = file_str.split('\n')
    new_file_str = ''
    word = ''

    for item in file_list:
        # if item != '':
        if len(item) >= 3:
            if item[2] == 'B':
                if word != '':
                    new_file_str += word + '\n'
                    word = item[0]
                else:
                    word += item[0]
            elif item[2] == 'O':
                if word != '':
                    new_file_str += word + '\n'
                    word = ''
            else:
                word += item[0]

    new_file = open(path + '_extracted.txt', 'w', encoding='utf-8')
    new_file.write(new_file_str)
