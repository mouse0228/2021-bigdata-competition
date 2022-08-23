# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:27:48 2021

@author: mouse0228
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import tqdm
import warnings 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU


warnings.filterwarnings('ignore')

data = pd.read_csv('train20210817v2.csv')
test = pd.read_csv('2021test0831.csv')

data.set_index('SeqNo',inplace=True)
test.set_index('SeqNo',inplace=True)
train = data[19611:26820]
train['F_1'][24151] = 14000
test['F_1'][4555] = 14000

train_O = train['O']

train.drop(train.columns[10:] ,axis=1, inplace=True)
test.drop(test.columns[10:] ,axis=1, inplace=True)

# 正規化
scaler = MinMaxScaler()
train['F_1'] = scaler.fit_transform(np.array(train['F_1']).reshape(-1,1))    
test['F_1'] = scaler.transform(np.array(test['F_1']).reshape(-1,1))    

train['F_2'] = train['F_2']-min(train['F_2'])
train['F_3'] = train['F_3']-min(train['F_3'])
train['F_4'] = train['F_4']-min(train['F_4'])
train['F_5'] = train['F_5']-min(train['F_5'])
train['F_6'] = train['F_6']-min(train['F_6'])
train['F_7'] = train['F_7']-min(train['F_7'])
train['F_8'] = train['F_8']-min(train['F_8'])
train['F_9'] = train['F_9']-min(train['F_9'])
train['F_10'] = train['F_10']-min(train['F_10'])

test['F_2'] = test['F_2']-min(test['F_2'])
test['F_3'] = test['F_3']-min(test['F_3'])
test['F_4'] = test['F_4']-min(test['F_4'])
test['F_5'] = test['F_5']-min(test['F_5'])
test['F_6'] = test['F_6']-min(test['F_6'])
test['F_7'] = test['F_7']-min(test['F_7'])
test['F_8'] = test['F_8']-min(test['F_8'])
test['F_9'] = test['F_9']-min(test['F_9'])
test['F_10'] = test['F_10']-min(test['F_10'])

 
n = 5 #改n即可，資料1/4起，所以能預測的第一個Y為2/3，抓30天
X1 = []
y1 = []  
testX = []

for i in tqdm.tqdm_notebook(range(0,len(train)-n)): 
    X1.append(train.iloc[i:i+n]. values) 
    y1.append(train_O.iloc[i+n-1]) #現有資料+30天的Y
for i in tqdm.tqdm_notebook(range(0,len(test)-n)):  
    testX.append(test.iloc[i:i+n]. values) #現有資料+30天的Y

X1=np.array(X1) 
y1=np.array(y1) 
testX=np.array(testX) 
  

n_steps = 5
n_features = 10

model = Sequential()
model.add(GRU(100,activation='relu', return_sequences=False, input_shape = (n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mae')
history = model.fit(X1,y1,batch_size=1000,epochs=10)
predictions = model.predict(testX)


data = pd.read_csv('train20210817v2.csv')
test = pd.read_csv('2021test0831.csv')
data.set_index('SeqNo',inplace=True)
test.set_index('SeqNo',inplace=True)
train = pd.concat([data[90864:91264],data[83653:84053],data[19611:20011]])

train_O = train['O']
 
train = train.drop(train.columns[2:12] ,axis=1, inplace=False)
test = test.drop(test.columns[2:12] ,axis=1, inplace=False)
train = train.drop(['O'] ,axis=1, inplace=False)
scaler = MinMaxScaler()
train['F_1'] = scaler.fit_transform(np.array(train['F_1']).reshape(-1,1))   
test['F_1'] = scaler.transform(np.array(test['F_1']).reshape(-1,1))    


train['F_2'] = train['F_2']-min(train['F_2'])
test['F_2'] = test['F_2']-min(test['F_2'])


err_model = RandomForestRegressor()
err_model.fit(train,train_O)
error = err_model.predict(test)
error = error.reshape(-1,1)


gap = sum(error[5:400]-predictions[:395])/len(error[5:400])
# plt.plot(error,color='black')
# plt.plot(predictions,color='g',label='pred') 
plt.plot(np.append(error[0:40],(predictions+gap)[35:]),color='r',label='pred+gap')
# plt.plot(np.linspace(1,500,500),data['O'][19611:20111],color='y',label='l1')
plt.plot(np.linspace(1,len(data['O'][19611:26820]),len(data['O'][19611:26820])),data['O'][19611:26820],color='y',label='l1')
plt.plot(np.linspace(1,len(data['O'][83653:90864]),len(data['O'][83653:90864])),data['O'][83653:90864],color='b',label='l2')
plt.plot(np.linspace(1,len(data['O'][90864:98072]),len(data['O'][90864:98072])),data['O'][90864:98072],color='g',label='l3')
plt.legend(loc='upper right')


gap = sum(error[5:400]-predictions[:395])/len(error[5:400])
plt.plot(error[0:400],color='black')
# plt.plot(predictions,color='g',label='pred') 
plt.plot(np.append(error[0:40],(predictions+gap)[35:395]),color='r',label='pred+gap')
# # plt.plot(np.linspace(1,500,500),data['O'][19611:20111],color='y',label='l1')
plt.plot(np.linspace(1,400,400),data['O'][19611:20011],color='y',label='l1')
plt.plot(np.linspace(1,400,400),data['O'][83653:84053],color='b',label='l2')
plt.plot(np.linspace(1,400,400),data['O'][90864:91264],color='g',label='l3')
plt.legend(loc='upper right')

a = np.append(error[0:40],(predictions+gap)[35:])

plt.plot(a)

result =  pd.read_excel('110056_TestResult.xlsx')
result['預測值'] = a

result.to_excel('110056_TestResult.xlsx',index=False)



