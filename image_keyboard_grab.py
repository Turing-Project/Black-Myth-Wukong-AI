# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:45:04 2020

@author: analoganddigital   ( GitHub )
"""

import numpy as np
import cv2
import time
import grabscreen
import getkeys
import os

wait_time = 5
L_t = 3
file_name = 'wukong_train.npy'
# window_size = (306,89,1029,569) # boss位置 # 1024*800分辨率 16:9
# window_size = (444,546,854,798) # 猴的倒地位置
window_size = (336,135,1395,795) # 1680*1050 分辨率 16:9

if os.path.isfile(file_name):
    print("file exists , loading previous data")
    training_data = list(np.load(file_name,allow_pickle=True))
else:
    print("file don't exists , create new one")
    training_data = []

for i in list(range(wait_time))[::-1]:
    print(i+1)
    time.sleep(1)

last_time = time.time()
while(True):
    output_key = getkeys.get_key(getkeys.key_check())#按键收集
    if output_key == [1,1,1,1,1,1,1,1,1,1]:
        print(len(training_data))
        training_data = np.array(training_data, dtype=object)
        np.save(file_name,training_data)
        break

    #printscreen = np.array(ImageGrab.grab(bbox=(window_size)))
    #printscreen_numpy = np.array(printscreen_pil.getdata(),dtype='uint8')\
    #.reshape((printscreen_pil.size[1],printscreen_pil.size[0],3))
    #pil格式耗时太长
    screen_rgb = grabscreen.grab_screen(window_size) # rgb图像收集
    screen_reshape = cv2.resize(screen_rgb,(175,200))
    
    training_data.append([screen_reshape,output_key])
    
    if len(training_data) % 500 == 0:
        print(len(training_data))
        arr = np.asarray(training_data, dtype = object)
        np.save(file_name,arr)
    cv2.imshow('window1',screen_rgb)
    #cv2.imshow('window3',printscreen)
    #cv2.imshow('window2',screen_reshape)
    #测试时间用
    print('loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cv2.waitKey()# 视频结束后，按任意键退出
cv2.destroyAllWindows()
