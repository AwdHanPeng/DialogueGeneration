from torch.utils.data import Dataset as DS
import pickle
import torch
from utils import *

'''
    def forward(self, src, tgt, src_mask, tgt_mask)
'''
'''
    文件之中自带头尾token
'''

class Dataset(DS):
    '''
    src:(batch_size, max_sentence_len)
    tgt:(batch_size, max_sentence_len-1)
    tgt_y:(batch_size,max_sentence_len-1)
    root 中包含三个文件路径
       src_path:应返回list(N,sentence_len)[未padding]
       tgt_path:应返回list(N,sentence_len)[未padding]
    '''

    def __init__(self, root, transform=None):
        src_path, tgt_path = root
        with open(src_path, 'rb') as f:
            self.src = pickle.load(f, encoding='bytes')
        with open(tgt_path, 'rb') as f:
            self.tgt = pickle.load(f, encoding='bytes')
        if transform:
            self.transform = transform

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = torch.tensor(self.transform(self.src[idx]))
        temp_tgt = torch.tensor(self.transform(self.tgt[idx]))
        tgt = temp_tgt[:-1].clone().detach()
        tgt_y = temp_tgt[1:].clone().detach()
        src_mask = torch.tensor([int(i != 0) for i in src]).unsqueeze(0)  # 有内容的是1 无内容的是0
        tgt_mask = torch.tensor([int(i != 0) for i in tgt])
        tgt_mask = subsequent_mask(len(tgt)).type_as(tgt_mask) & tgt_mask  # 有内容的是1 无内容的是0
        return src, tgt, tgt_y, src_mask, tgt_mask

class TestDataset(DS):
    '''
        encoder输入 decoder输入是一个一个变长的 所以不需要
        decoder输出需要一个完整的对标数据
        mask
    '''

    def __init__(self, root, transform=None):
        src_path, tgt_path = root
        with open(src_path, 'rb') as f:
            self.src = pickle.load(f, encoding='bytes')
        with open(tgt_path, 'rb') as f:
            self.tgt = pickle.load(f, encoding='bytes')
        if transform:
            self.transform = transform

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = torch.tensor(self.transform(self.src[idx]))
        temp_tgt = torch.tensor(self.transform(self.tgt[idx]))
        tgt = temp_tgt[:-1].clone().detach()
        src_mask = torch.tensor([int(i != 0) for i in src]).unsqueeze(0)
        tgt_mask = torch.tensor([int(i != 0) for i in tgt])
        tgt_mask = subsequent_mask(len(tgt)).type_as(tgt_mask) & tgt_mask
        return src, temp_tgt, src_mask, tgt_mask
