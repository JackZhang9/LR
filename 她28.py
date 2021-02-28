#!/usr/bin/env python
# _*_coding: utf-8 _*_
# @Time : 2021/2/28 10:46
# @Author : CN-JackZhang
# @File: 她28.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)
#数据导入
df = pd.read_csv(r'D:\python\python38\lib\site-packages\sklearn\datasets\data\breast_cancer.csv')
Y = df.pop('label').values  #一维数组
X = df.values               #二维数组
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=42)
#sigmoid,[0,1]
def sigmoid(X):
    return 1/(1+np.exp(-X))
#训练w
def fit(X_train,Y_train):
    rows,columns = X_train.shape
    w = np.ones(columns)
    # print('wtye',w,type(w))
    alpha = 0.001
    loss = np.zeros(columns)
    # print(loss)
    for k in range(50):
        for j in range(rows):
            # print('Xtye',X_train[j],type(X_train[j]))
            # print(np.dot(w,X_train[j]))
            loss += (sigmoid(np.dot(w,X_train[j])) - Y_train[j])*X_train[j]
        w -= alpha*loss
        # print('第{}次，w值得变化{}'.format(k,w))
    return w

w = fit(X_train,Y_train)
#预测
def predict(X_test):
    rows,columns = X_test.shape
    predict = np.empty(rows)
    for i in range(rows):
        predict [i] = sigmoid(np.dot(w,X_test[i]))
    return predict
predict = predict(X_test)
# w = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
# print(sigmoid(np.dot(w,X_test[1])),Y_test[1])
# print(sigmoid(np.dot(w,X_test[2])),Y_test[2])
# print(sigmoid(np.dot(w,X_test[3])),Y_test[3])
# print(sigmoid(np.dot(w,X_test[4])),Y_test[4])
# print(sigmoid(np.dot(w,X_test[5])),Y_test[5])
# print(sigmoid(np.dot(w,X_test[6])),Y_test[6])
# print(sigmoid(np.dot(w,X_test[7])),Y_test[7])
# print(sigmoid(np.dot(w,X_test[8])),Y_test[8])
# print(sigmoid(np.dot(w,X_test[9])),Y_test[9])

print(accuracy_score(Y_test,predict))



# print(X_train,Y_train)

# print(df.head())