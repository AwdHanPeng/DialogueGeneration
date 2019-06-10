import pickle
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

PAD_token = 0
BEG_token = 1
EOS_token = 2
with open('../data/index2word.pkl', 'rb') as f:
    index2word = pickle.load(f, encoding='bytes')

def getlist(path):
    with open(path, 'rb') as f:
        sentences = pickle.load(f, encoding='bytes')
    for i, sentence in enumerate(sentences):
        if sentence.count(EOS_token) is not 0:
            idx = sentence.index(EOS_token)
            temp = sentence[:idx]
        else:
            temp = sentence
        while temp.count(PAD_token) is not 0:
            temp.remove(PAD_token)
        while temp.count(BEG_token) is not 0:
            temp.remove(BEG_token)
        while temp.count(EOS_token) is not 0:
            temp.remove(EOS_token)
        temp = [index2word[i] for i in temp]
        sentences[i] = temp
    return sentences

if __name__ == '__main__':
    src = getlist('../data/test/test_src.pkl')
    tgt = getlist('../data/test/test_tgt.pkl')
    gen = getlist('../data/test/test_gen.pkl')
    assert len(src) == len(tgt)
    print('case sum is {}'.format(len(gen)))
    bleu = 0
    smoothing = SmoothingFunction()
    with open('../data/test/case.txt', 'w', encoding='UTF-8') as f:
        for i, (s, t, g) in enumerate(zip(src, tgt, gen)):
            # print('source:{}\ntarget:{}\ngeneration:{}\n'.format(s, t, g))
            bleu += sentence_bleu(t, g, weights=[1], smoothing_function=smoothing.method0)
            if i % 1000 == 0:
                print('Cac at {} samples'.format(i))
            f.write('source:{}\n'.format(s))
            f.write('target:{}\n'.format(t))
            f.write('generation:{}\n'.format(g))
            f.write('\n')

    print('BLEU is {:2f}'.format(bleu / len(gen)))
