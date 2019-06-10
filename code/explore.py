'''
    此文件用于探索样本规模
'''

def explore(file):
    with open(file, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        sum_sentence = len(lines)
        print(sum_sentence)
        max_len = 0
        sum_sentence_len = 0
        for line in lines:
            tempsente = line.replace('\n', '').split(' ')
            sum_sentence_len += len(tempsente)
            if len(tempsente) > max_len:
                max_len = len(tempsente)
        print('平均每句话几个单词{}'.format(sum_sentence_len / sum_sentence))
        print('max is {}'.format(max_len))

if __name__ == '__main__':
    explore('../data/train_src.data')  # 1000000 平均每句话几个单词11.809365
    explore('../data/train_tgt.data')  # 1000000 平均每句话几个单词9.56969
