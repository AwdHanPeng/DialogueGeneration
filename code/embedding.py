import pickle
import random

sum = 8824330

def getword2index():
    with open('../data/word2index.pkl', 'rb') as f:
        word2index = pickle.load(f)
    return word2index

def create_embedding(word2index, src_path):
    embedding = dict()
    with open(src_path, 'r', encoding='UTF-8') as f:
        _ = f.readline()
        for i in range(sum):
            line = f.readline()
            if i % 100000 == 0:
                print('create map at {} word in tencent'.format(i + 1))
            temp = line.replace('\n', '').split(' ')
            if temp[0] in word2index:
                embedding[word2index[temp[0]]] = [float(i) for i in temp[1:]]
            if len(embedding) == len(word2index) - 2:
                break
    print('complete to get embedding')
    count = 0
    output = []
    for i in range(len(word2index)):
        if i not in embedding:
            output.append([random.random() - 0.5 for i in range(200)])
            count += 1
        else:
            output.append(embedding[i])
    print('{} word do not appear in tencent_embedding'.format(count - 1))
    with open('../data/embedding/embedding.pkl', 'wb') as f:
        pickle.dump(output, f)
    print('complete to create embedding')

if __name__ == '__main__':
    word2index = getword2index()
    create_embedding(word2index, 'C://embedding//Tencent.txt')
