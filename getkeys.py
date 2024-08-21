# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 12:03:44 2020

@author: analoganddigital   ( GitHub )
"""

import win32api as wapi
import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys

def get_key(keys):
    '''
    #W,A,S,D,V,J,M,K
    output = [0,0,0,0,0,0,0,0]
    if 'W' in keys:
        output[0] = 1
    elif 'A' in keys:
        output[1] = 1
    elif 'S' in keys:
        output[2] = 1
    elif 'D' in keys:
        output[3] = 1
    elif 'V' in keys:
        output[4] = 1
    elif 'J' in keys:
        output[5] = 1
    elif 'M' in keys:
        output[6] = 1
    elif 'K' in keys:
        output[7] = 1 
    elif '8' in keys:#停止记录操作
        output = [1,1,1,1,1,1,1,1]
    '''
    '''
    #W,J,M,K,none
    output = [0,0,0,0,0]
    if 'W' in keys:
        output[0] = 1
    elif 'J' in keys:
        output[1] = 1
    elif 'M' in keys:
        output[2] = 1
    elif 'K' in keys:
        output[3] = 1 
    elif '8' in keys:#停止记录操作
        output = [1,1,1,1]
    '''   
    
    #W,J,M,K,R,none          R代替左shift，识破
    output = [0,0,0,0,0,0]
    if 'W' in keys:
        output[0] = 1
    elif 'J' in keys:
        output[1] = 1
    elif 'M' in keys:
        output[2] = 1
    elif 'K' in keys:
        output[3] = 1 
    elif 'R' in keys:
        output[4] = 1 
    elif '8' in keys:#停止记录操作
        output = [1,1,1,1,1,1]
    else:
        output[5] = 1
        
    return output

if __name__ == '__main__':
    while True:
        if get_key(key_check()) != [1,1,1,1,1,1]:
            print(key_check())
        else:
            break