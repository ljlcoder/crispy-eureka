#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :split.py
# @Time      :2024/10/18 16:06
# @Author    :Jasonljl

from sklearn.model_selection import train_test_split
from read_csv import readcsv,transfomer
from torch.utils.data import Dataset,DataLoader
import torch
data=readcsv("./train_set.csv")
x,y=transfomer(data)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


if __name__ == '__main__':
    print(x_train)