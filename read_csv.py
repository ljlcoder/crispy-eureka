#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :read_csv.py
# @Time      :2024/10/18 16:01
# @Author    :Jasonljl
import pandas as pd


def readcsv(path):
    data=pd.read_csv(path)
    return data
def transfomer(df):
    #reshape(-1,1)是对一个特征用的
    #x=df.iloc[:,1].values.reshape(-1，1)
    x=df.iloc[:,:2].values
    y=df.iloc[:,6].values
    return x,y
if __name__ == '__main__':
    data=readcsv("./test_set.csv")
    print(data.columns)
    x,y=transfomer(data)
    print(x.shape,y.shape)