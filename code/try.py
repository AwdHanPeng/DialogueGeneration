import pickle
import matplotlib.pylab as plt
import torch

def imshow(obj, title):
    epoch = [i['epoch'] for i in obj]
    loss = [i['loss'] for i in obj]
    plt.figure()
    plt.plot(epoch, loss, label='LOSS', color="#EE00EE")
    plt.title(title)
    plt.legend(loc='upper right')
    plt.xticks(epoch)
    plt.grid(alpha=0.4, linestyle=':')
    plt.show()

if __name__ == '__main__':
    with open('../model/history.pkl', 'rb') as f:
        history = pickle.load(f)
        print(history)
