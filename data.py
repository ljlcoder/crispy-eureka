#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :data.py
# @Time      :2024/10/18 16:14
# @Author    :Jasonljl
from torch.utils.data import Dataset,DataLoader
import torch
from split import *
x_train_tensor=torch.from_numpy(x_train.astype("float32"))
x_test_tensor=torch.from_numpy(x_test.astype("float32"))
y_train_tensor=torch.from_numpy(y_train.astype("float32"))
y_test_tensor=torch.from_numpy(y_test.astype("float32"))

class CustomDataset(Dataset):
    def __init__(self,features,labels):
        self.features=features
        self.labels=labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, item):
        return self.features[item],self.labels[item]
train_dataset=CustomDataset(x_train_tensor,y_train_tensor)
test_dataset=CustomDataset(x_test_tensor,y_test_tensor)

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(train_dataset,batch_size=32,shuffle=False)

if __name__ == '__main__':
    print(train_loader)
