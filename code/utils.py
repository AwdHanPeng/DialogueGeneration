import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pylab as plt
from nltk.translate.bleu_score import sentence_bleu
import pickle
from torch import nn

PAD_token = 0
BEG_token = 1
EOS_token = 2

src_vocab = 93379
tgt_vocab = 93379
max_sentence_len = 15

# 19150 word do not appear in tencent_embedding
def padding_sentence(sentence, max_sentence_len=max_sentence_len):
    while len(sentence) < max_sentence_len:
        sentence.append(PAD_token)
    while len(sentence) > max_sentence_len:
        sentence.pop(-2)  # 保留住尾部的EOS_token
    return sentence

def subsequent_mask(size):
    attn_shape = (size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = LabelSmoothing(size=src_vocab, padding_idx=PAD_token, smoothing=0.1)
    criterion.to(device)
    for batch_idx, (src, tgt, tgt_y, src_mask, tgt_mask) in enumerate(train_loader):
        src, tgt, tgt_y, src_mask, tgt_mask = map(lambda x: x.to(device), [src, tgt, tgt_y, src_mask, tgt_mask])
        optimizer.optimizer.zero_grad()
        output = model(src, tgt, src_mask, tgt_mask)
        prob = model.generator(output)
        loss = criterion(prob.view(-1, src_vocab), tgt_y.view(-1))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, batch_idx * len(src), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def valid(args, model, device, valid_loader):
    model.eval()
    average_loss = 0
    criterion = LabelSmoothing(size=src_vocab, padding_idx=PAD_token, smoothing=0.1)
    criterion.to(device)
    with torch.no_grad():
        for batch_idx, (src, tgt, tgt_y, src_mask, tgt_mask) in enumerate(valid_loader):
            src, tgt, tgt_y, src_mask, tgt_mask = map(lambda x: x.to(device), [src, tgt, tgt_y, src_mask, tgt_mask])
            output = model(src, tgt, src_mask, tgt_mask)
            prob = model.generator(output)
            loss = criterion(prob.view(-1, src_vocab), tgt_y.view(-1))
            average_loss += loss
            if batch_idx % args.log_interval == 0:
                print('Valid at sample {} [{}/{} ({:.0f}%)\tLoss: {:.6f}\t]'.format(batch_idx * args.valid_batch_size,
                                                                                    batch_idx * args.valid_batch_size,
                                                                                    len(valid_loader.dataset),
                                                                                    100. * batch_idx * args.valid_batch_size / len(
                                                                                        valid_loader.dataset),
                                                                                    loss.item()))
        average_loss /= len(valid_loader)
        print('Validation Average Loss: {:.6f}\t'.format(average_loss))
    return average_loss.item()

def test(args, model, device, test_loader):
    model.eval()
    generation = []
    with torch.no_grad():
        for batch_idx, (src, tgt, src_mask, tgt_mask) in enumerate(test_loader):
            src, tgt, src_mask, tgt_mask = map(lambda x: x.to(device), [src, tgt, src_mask, tgt_mask])
            memory = model.encode(src, src_mask)
            ys = torch.ones(args.test_batch_size, 1).fill_(BEG_token).type_as(src.data)
            for i in range(max_sentence_len - 1):
                out = model.decode(memory, src_mask, ys.to(device),
                                   subsequent_mask(ys.size(1)).unsqueeze(0).type_as(src.data).to(device))
                # out: N=1,L,D_model
                prob = model.generator(out)[:, -1, :]
                # prob: N=1,[final],num_words ->num_words
                _, next_word = torch.max(prob, dim=-1)
                ys = torch.cat((ys, next_word.unsqueeze(-1)), dim=-1)  # N*1->N*2->N*3
            generation.extend(ys.tolist())
            if batch_idx >= 1000:
                break
            if batch_idx % args.log_interval == 0:
                print('Generate Sentence at {} samples'.format(batch_idx * args.test_batch_size))
        return generation

class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)
