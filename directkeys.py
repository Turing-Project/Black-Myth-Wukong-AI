'''
Description: 
Version: 2.0
Autor: Zhang
Date: 2021-11-14 15:08:58
LastEditors: Zhang
LastEditTime: 2021-12-04 15:02:57
'''

import ctypes
import time

SendInput = ctypes.windll.user32.SendInput


W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

M = 0x32
J = 0x24
K = 0x25
LSHIFT = 0x2A
R = 0x13#用R代替识破
V = 0x2F

Q = 0x10
I = 0x17
O = 0x18
P = 0x19
C = 0x2E
F = 0x21

up = 0xC8
down = 0xD0
left = 0xCB
right = 0xCD

esc = 0x01

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
    
def defense_RL(t): #持续防御_强化学习
    PressKey(M)
    time.sleep(t)
    ReleaseKey(M)
    #time.sleep(0.1)

def defense():   #格挡
    PressKey(M)
    time.sleep(0.05)
    ReleaseKey(M)
    time.sleep(0.15)
    
def attack():   #攻击
    PressKey(J)
    time.sleep(0.05)
    ReleaseKey(J)
    time.sleep(0.6)

def hard_attack(): #突刺重攻击
    PressKey(J)
    time.sleep(0.7)
    ReleaseKey(J)
    time.sleep(0.4)

def ninja_attack(): #忍义手攻击
    PressKey(Q)
    time.sleep(0.05)
    ReleaseKey(Q)
    time.sleep(0.5)
    # PressKey(Q)
    # time.sleep(0.05)
    # ReleaseKey(Q)
    # time.sleep(0.5)

def skill_attack(): #技能
    PressKey(J)
    PressKey(M)
    time.sleep(0.05)
    ReleaseKey(J)
    ReleaseKey(M)
    time.sleep(0.9)
    # PressKey(J)
    # time.sleep(0.05)
    # ReleaseKey(J)
    # time.sleep(1.9)
    
def go_forward(): #前进
    PressKey(W)
    time.sleep(0.4)
    ReleaseKey(W)
    
def go_back(): #后退
    PressKey(S)
    time.sleep(0.4)
    ReleaseKey(S)
    
def go_left(): #左
    PressKey(A)
    time.sleep(0.4)
    ReleaseKey(A)
    
def go_right(): #右
    PressKey(D)
    time.sleep(0.4)
    ReleaseKey(D)
    
def jump():  #跳跃
    PressKey(K)
    time.sleep(0.1)
    ReleaseKey(K)
    time.sleep(0.1)
    
def dodge_forward():#冲刺
    PressKey(R)
    time.sleep(0.1)
    ReleaseKey(R)
    time.sleep(0.4)

def dodge_back():#后撤
    PressKey(S)
    time.sleep(0.25)
    PressKey(R)
    time.sleep(0.05)
    ReleaseKey(R)
    ReleaseKey(S)
    time.sleep(0.3)

def dodge_right():# 右闪
    PressKey(D)
    time.sleep(0.25)
    PressKey(R)
    time.sleep(0.05)
    ReleaseKey(R)
    ReleaseKey(D)
    time.sleep(0.3)

def dodge_left():# 左闪
    PressKey(A)
    time.sleep(0.25)
    PressKey(R)
    time.sleep(0.05)
    ReleaseKey(R)
    ReleaseKey(A)
    time.sleep(0.3)

def use_items(): #使用道具
    PressKey(R)
    time.sleep(0.1)
    ReleaseKey(R)

    
def lock_vision(): #锁定敌人
    PressKey(V)
    time.sleep(0.3)
    ReleaseKey(V)
    time.sleep(0.1)
    
def go_forward_RL(t): #持续前进_强化学习
    PressKey(W)
    time.sleep(t)
    ReleaseKey(W)
    
def turn_left(t): #视角向左
    PressKey(left)
    time.sleep(t)
    ReleaseKey(left)
    
def turn_up(t): #视角向上
    PressKey(up)
    time.sleep(t)
    ReleaseKey(up)
    
def turn_right(t): #视角向右
    PressKey(right)
    time.sleep(t)
    ReleaseKey(right)
    
def F_go():  #钩爪
    PressKey(F)
    time.sleep(0.5)
    ReleaseKey(F)
    
def forward_jump(t):  #向前跳跃
    PressKey(W)
    time.sleep(t)
    PressKey(K)
    ReleaseKey(W)
    ReleaseKey(K)
    
def press_esc():  #菜单
    PressKey(esc)
    time.sleep(0.3)
    ReleaseKey(esc)
    
def dead():  # 死亡(不回生)
    PressKey(M)
    time.sleep(0.5)
    ReleaseKey(M)

def revive():  # 回生
    PressKey(J)
    time.sleep(0.5)
    ReleaseKey(J)

def feidu():
    defense()
    defense()
    defense()
    defense()
    defense()
    # defense()
    # defense()
    # defense()
    # defense()
    # defense()
    # defense()
    # defense()
    
if __name__ == '__main__':
    time.sleep(5)
    time1 = time.time()
    while(True):
        if abs(time.time()-time1) > 5:
            break
        else:
            PressKey(M)
            time.sleep(0.1)
            ReleaseKey(M)
            time.sleep(0.2)
        
    attack()
    dodge_forward()
    attack()
    dodge_forward()
    attack()
    attack()
    #skill_attack()
    #attack()
    defense()

   # attack()
   #attack()
   # attack()
    #go_forward()
    #time.sleep(0.5)
    #skill_attack()
    #time.sleep(0.5)
    #skill_attack()
    #time.sleep(0.5)
    #skill_attack()
    #time.sleep(0.5)
    #dodge_forward()
    #time.sleep(0.5)
    #dodge_back()
    #time.sleep(0.5)
    #hard_attack()
    #time.sleep(0.5)
    #ninja_attack()