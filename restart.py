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
        # time.sleep(3)
        print("开始新一轮")
        # pyautogui.keyDown('num2')
        # pyautogui.keyDown('num2')
        # pyautogui.keyDown('num2') # 必须要3次它才能检测到，原因未知，少一次都不行
        # time.sleep(1)
        # pyautogui.keyUp('num2') # 释放按键，下一次才能正确按到
        # pass
    else :
        pass
  
if __name__ == "__main__":  
    restart()