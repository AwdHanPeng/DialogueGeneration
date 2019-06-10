from model import make_model
import argparse
import torch
import torch.optim as optim
from dataset import TestDataset
from utils import *
import pickle

parser = argparse.ArgumentParser(description='Test')

parser.add_argument('--test-batch-size', type=int, default=80,
                    help='input batch size for training (default: 1)')
parser.add_argument('--epochs', type=int, default=1,
                    help='number of epochs to train (default: 1)')
parser.add_argument('--no-cuda', default=0, type=int,
                    help='disables CUDA training')
parser.add_argument('--img', default=0, type=int,
                    help='show image')
parser.add_argument('--log-interval', type=int, default=5,
                    help='how many batches to wait before logging training or validation status')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
test_root = ['test_src', 'test_tgt']
test_root_path = ['../data/test/' + i + '.pkl' for i in test_root]

test_loader = torch.utils.data.DataLoader(
    TestDataset(root=test_root_path, transform=padding_sentence),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)

if __name__ == '__main__':
    best_model = 'model10'
    model = make_model(src_vocab, tgt_vocab)
    model.load_state_dict(
        torch.load('../model/{}.pt'.format(best_model))
    )
    model = model.to(device)
    generation = test(args, model, device, test_loader)
    print('It is time to save generation sentences')
    with open('../data/test/test_gen.pkl', 'wb') as f:
        pickle.dump(generation, f)
    if args.img:
        with open('../model/history.pkl', 'wb') as f:
            history_list = pickle.load(f, encoding='bytes')
            imshow(history_list, 'History')
