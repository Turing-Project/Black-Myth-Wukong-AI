# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:28:11 2020

@author: analoganddigital   ( GitHub )
"""

import os
import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

file_name = 'wukong_train.npy'
if os.path.isfile(file_name):
    print("file exists , loading previous data")
    training_data = list(np.load(file_name,allow_pickle=True))

''' 
w=[]
a=[]
s=[]
d=[]
v=[]
j=[]
m=[]
k=[]
n_choice=[]
'''

data_1=[]
data_2=[]
data_3=[]
data_4=[]
data_5=[]
data_6=[]
data_7=[]
data_8=[]
data_9=[]
data_0=[]
data_z=[]
n_choice=[]

df = pd.DataFrame(training_data)
print(df.head())
print(Counter(df[1].apply(str)))    

for data in training_data:                  
    img = data[0]
    choice = data[1]
    if choice[0] == 1:
        data_1.append([img,choice])
    elif choice[1] == 1:
        data_3.append([img,choice])
    elif choice[2] == 1:
        data_4.append([img,choice])
    elif choice[3] == 1:
        data_5.append([img,choice])
    elif choice[4] == 1:
        data_6.append([img,choice])
    elif choice[5] == 1:
        data_7.append([img,choice])
    elif choice[6] == 1:
        data_8.append([img,choice])
    elif choice[7] == 1:
        data_0.append([img,choice])
    elif choice[8] == 1:
        data_z.append([img,choice])
    elif choice[9] == 1:
        data_9.append([img,choice])

    
# Find the label with the maximum number of data points
max_length = max(len(data_1),len(data_3), len(data_4), len(data_5),
                 len(data_6), len(data_7), len(data_8), len(data_9), len(data_0),
                 len(data_z))

# max_length = 1500

data_1 = data_1[:max_length]
data_3 = data_3[:max_length]
data_4 = data_4[:max_length]
data_5 = data_5[:max_length]
data_6 = data_6[:max_length]
data_7 = data_7[:max_length]
data_8 = data_8[:max_length]
data_0 = data_0[:max_length]
data_9 = data_9[:max_length]
data_z = data_z[:max_length]


final_data = data_1+  data_3+ data_4+ data_5+ data_6+ data_7+ data_8+ data_9+ data_0+ data_z
shuffle(final_data)
print(len(final_data))
# 用asarray避免维数不同报错
arr = np.asarray(final_data, dtype = object)
np.save('wukong_train_new_daolang_5.npy',arr)


'''
for data in training_data:
    img = data[0]
    choice = data[1]
    cv2.imshow('test',img)
    print(choice)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cv2.waitKey()# 视频结束后，按任意键退出
cv2.destroyAllWindows()
'''

df = pd.DataFrame(final_data)
print(df.head())
print(Counter(df[1].apply(str)))

