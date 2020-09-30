import Data

f = open(r'NERData\MSRA\msra_test_bio.txt', 'r', encoding='utf-8')
file = f.read()
file = file.split('\n')
sentences = []
max_length = 0
max_index = 0
sentence = []
length = 0
count = -1
m_index = 0
for i, item in enumerate(file):
    if item != '':
        sentence.append(item)
        length += 1
    elif item == '' and length > 0:
        sentences.append(sentence)
        count += 1
        if length > max_length:
            max_length = length
            max_index = count
            m_index = i
        sentence = []
        length = 0
    else:
        continue

print(sentences[max_index])
print(m_index)
