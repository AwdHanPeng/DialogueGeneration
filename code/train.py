from model import make_model
import argparse
import torch
import torch.optim as optim
from dataset import Dataset
from utils import *
import pickle

parser = argparse.ArgumentParser(description='Train')

parser.add_argument('--batch-size', type=int, default=100,
                    help='input batch size for training (default: 200)')
parser.add_argument('--valid-batch-size', type=int, default=100,
                    help='input batch size for validation (default: 200)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta 1 (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.98,
                    help='beta 2 (default: 0.999)')
parser.add_argument('--eps', type=float, default=1e-9,
                    help='eps (default: 1e-9)')
parser.add_argument('--no-cuda', default=0, type=int,
                    help='disables CUDA training')
parser.add_argument('--save-model', default=1, type=int,
                    help='For Saving the current Model')
parser.add_argument('--log-interval', type=int, default=5,
                    help='how many batches to wait before logging training or validation status')
parser.add_argument('--toleration', type=int, default=2,
                    help='Stop training after several drops')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_root = ['train_src', 'train_tgt']
train_root_path = ['../data/train/' + i + '.pkl' for i in train_root]
valid_root = ['valid_src', 'valid_tgt']
valid_root_path = ['../data/valid/' + i + '.pkl' for i in valid_root]

train_loader = torch.utils.data.DataLoader(
    Dataset(root=train_root_path, transform=padding_sentence),
    batch_size=args.batch_size, shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(
    Dataset(root=valid_root_path, transform=padding_sentence),
    batch_size=args.valid_batch_size, shuffle=False, **kwargs)

if __name__ == '__main__':
    history_list = []
    model = make_model(src_vocab, tgt_vocab)
    optimizer = NoamOpt(model.src_embed[0].d_model, 1, 4000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    model = model.to(device)
    best_loss = float('Inf')
    last = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        loss = valid(args, model, device, valid_loader)
        history_list.append({
            'epoch': epoch,
            'loss': loss
        })
        if args.save_model and loss < best_loss:
            torch.save(model.state_dict(), "../model/model{}.pt".format(epoch))
            print('save model when epoch = {} as model{}.pt'.format(epoch, epoch))
            best_loss = loss
            best_epo = epoch
        if loss > best_loss:
            last += 1
        if last >= args.toleration:
            break
    print('best loss = {} at epoch {}'.format(best_loss, best_epo))
    with open('../model/history.pkl', 'wb') as f:
        pickle.dump(history_list, f)
