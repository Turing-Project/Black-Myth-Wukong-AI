'''
Description: 
Version: 2.0
Autor: Zhang
Date: 2021-11-14 17:29:46
LastEditors: Zhang
LastEditTime: 2021-12-02 10:14:29
'''
import directkeys as directkeys
import time

def restart(initial = False):
    if initial == False:
        print("死,restart")
        time.sleep(8)
        directkeys.lock_vision()
        time.sleep(0.2)
        directkeys.attack()
        print("开始新一轮")
    else :
        pass
  
if __name__ == "__main__":  
    restart()