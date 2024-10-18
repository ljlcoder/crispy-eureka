#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :model.py
# @Time      :2024/10/18 16:22
# @Author    :Jasonljl
import torch.optim

from data import *
import torch.nn as nn
from tqdm import tqdm
class Simplemodel(nn.Module):
    def __init__(self):
        super(Simplemodel,self).__init__()
        self.features=nn.Sequential(
            nn.Linear(2, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self,x):
        return self.features(x)

model=Simplemodel()
criterion=torch.nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)
if __name__ == '__main__':

    for epoch in range(10):
        for inputs,labels in tqdm(train_loader):
            optimizer.zero_grad()
            ouputs=model(inputs)
            loss=criterion(ouputs,labels)
            # print(f"epoch:{epoch}/{100},loss:{loss}")
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        totol_loss=0
        for inputs,labels in test_loader:
            ouputs=model(inputs)
            loss=criterion(ouputs,labels)
            totol_loss+=loss.item()
            # print(f"totol_loss:{totol_loss}")
        avg_loss=totol_loss/len(test_loader)
        print(f"test loss:{avg_loss}")