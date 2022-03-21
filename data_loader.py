import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import pandas as pd
import joblib
import pickle
import os
import random


# # 方法1，全量加载
class MyDataset(Dataset):
    def __init__(self, path='', tokenizer=None, batch_size=64, random_state=42, sce='train'):
        # 如果是debug模式可以只取一小部分数据，跑通整个流程以后再上全量
        if sce == 'debug':
            self.data = joblib.load('data.pkl')[:batch_size * 3]
        else:
            # 加载数据
            self.data = [random.random() for _ in range(64)]
            self.label = [1 for _ in range(64)]
            # # joblib加载
            # self.data = joblib.load(path)
            # # pickle加载
            # self.data = pickle.load(open(path,'rb'))
        self.batch_size = batch_size
        self.random_state = random_state
        self.sce = sce
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        assert idx < len(self.data)
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        # 这里的batch一般是根据batchsize调用__getitem__返回的list
        # 如果__getitem__返回是tuple，这里需要把输入和label分开
        labels = []
        inputs = []
        for sample, label in batch:
            inputs.append(sample)
            labels.append(label)
        # 对batch预处理
        # ...
        inputs = self.tokenizer.pad(inputs)
        return inputs, labels


# # 方法2，大数据分批次加载
class MyIterableDataset(IterableDataset):
    def __init__(self, path='', tokenizer=None, batch_size=64, random_state=42, sce='train'):
        # 如果是debug模式可以只取一小部分数据，跑通整个流程以后再上全量
        if sce == 'debug':
            self.data = joblib.load('data.pkl')[:batch_size * 3]
        else:
            # 加载数据
            self.data_iter = [[random.random() for _ in range(1)] for _ in range(64)]
            self.label_iter = [[1 for _ in range(1)] for _ in range(64)]
            # # panda存储的：先处理存成按一条信息存储的pandas，然后利用chunksize读取
            # self.data_iter=pd.read_csv(path,encoding='gb18030',chunksize=1) #注意这里直接用chunksize=1
            # # pkl存储的：存储的时候用'ab'模型是逐条dump，堵取的时候用while True+try读取
            # self.data_iter = open(path, 'rb')

        self.batch_size = batch_size
        self.random_state = random_state
        self.sce = sce
        self.tokenizer = tokenizer

    def __iter__(self):
        for data, labels in zip(self.data_iter, self.label_iter):
            random.shuffle(data)
            # 对batch预处理
            # ...
            # self.tokenizer.pad(batch)
            yield data, labels


class MyDataLoader(DataLoader):
    def __init__(self, data_path='', tokenizer=None, batch_size=1, random_state=42, sce='train', shuffle=False,
                 sampler=None, batch_sampler=None, num_workers=0, pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None):
        dataset = MyDataset(data_path, tokenizer, batch_size, random_state, sce)
        super(MyDataLoader, self).__init__(dataset=dataset,
                                           batch_size=dataset.batch_size,
                                           shuffle=shuffle,
                                           sampler=sampler,
                                           batch_sampler=batch_sampler,
                                           num_workers=num_workers,
                                           collate_fn=dataset.collate_fn,
                                           pin_memory=pin_memory,
                                           drop_last=drop_last,
                                           timeout=timeout,
                                           worker_init_fn=worker_init_fn,
                                           multiprocessing_context=multiprocessing_context)


class MyIterableDataLoader(DataLoader):
    def __init__(self, data_path='', tokenizer=None, batch_size=1, random_state=42, sce='train', shuffle=False,
                 sampler=None, batch_sampler=None, num_workers=0, pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None):
        dataset = MyIterableDataset(data_path, tokenizer, batch_size, random_state, sce)
        super(MyIterableDataLoader, self).__init__(dataset=dataset,
                                                   batch_size=dataset.batch_size,
                                                   shuffle=shuffle,
                                                   sampler=sampler,
                                                   batch_sampler=batch_sampler,
                                                   num_workers=num_workers,
                                                   pin_memory=pin_memory,
                                                   drop_last=drop_last,
                                                   timeout=timeout,
                                                   worker_init_fn=worker_init_fn,
                                                   multiprocessing_context=multiprocessing_context)


if __name__ == '__main__':
    # 分开Dataset和DataLoader
    dataset = MyDataset(path='', tokenizer=None, batch_size=64, random_state=42, sce='train')
    dataset_iter = MyIterableDataset(path='', tokenizer=None, batch_size=64, random_state=42, sce='train')
    # 构建DataLoader
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    dataloader_iter = DataLoader(dataset_iter, batch_size=64)

    # 直接包装成DataLoader
    dataloader = MyDataLoader(data_path='', tokenizer=None, batch_size=64, random_state=42, sce='train')

    dataloader_iter = MyIterableDataLoader(data_path='', tokenizer=None, batch_size=64, random_state=42, sce='train')
