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
import pyautogui
def restart(initial = False):
    if initial == False:
        print("死,restart")
        print("开始新一轮")
        time.sleep(3)
        # 以下用风灵月影满血以增加训练效率
        pyautogui.keyDown('num2')
        pyautogui.keyDown('num2')
        pyautogui.keyDown('num2') 
        time.sleep(1)
        pyautogui.keyUp('num2') 
        # pass
    else :
        pass
  
if __name__ == "__main__":  
    restart()