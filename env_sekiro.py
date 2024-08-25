import numpy as np
import torch
from grabscreen import grab_screen
import cv2
import time
import directkeys as directkeys
from getkeys import key_check
from restart import restart
import matplotlib.pyplot as plt
import pyautogui
class Sekiro(object):
    def __init__(self, observation_w, observation_h, action_dim):
        super().__init__()

        self.observation_dim = observation_w * observation_h
        self.width = observation_w
        self.height = observation_h

        self.action_dim = action_dim

        # self.obs_window = (379,160,675,500) # 显示器
        self.obs_window =  (306,89,1029,569)# 笔记本
        
        # 351 66 683 530
        # 399 211 651 480
        # self.obs_window = (400,120,669,471)
        # self.obs_window = (571,116,1075,766) #显示器 1600*900
        # self.blood_window = (104,128,621,863) #2560*1600笔记本屏幕分辨率，为--画质下
        # self.blood_window = (121,130,738,1013) #我的2560*1440显示器分辨率 为1920*1080画质下 捏吗这个卡昏了 别用
        # self.blood_window = (68,84,398,554)  #显示器1024*576
        # self.boss_blood_window = (71,95,287,99)  #笔记本1024*576
        self.boss_blood_window = (510,692,780,702)# 黑神话
        # self.sekiro_blood_window = (75,567,396,570)  #笔记本1024*576
        self.sekiro_blood_window = (144,741,254,746)  #黑神话 
        # self.stamina_window = (351,66,683,530) #显示器
        # self.blood_window = (103,113,605,848)  # 显示器 1600*900
        self.boss_stamina_window = (345,78,690,81) #笔记本
        # self.sekiro_stamina_window = (426,542,626,545) #笔记本
        self.sekiro_stamina_window = (1128,725,1158,779) # 黑神话
        # self.stamina_window = (549,86,1074,88)  # 显示器1600*900`

        self.boss_blood = 0
        self.self_blood = 0
        self.boss_stamina = 0
        self.self_stamina = 0

        self.stop = 0
        self.emergence_break = 0

    def self_blood_count(self,obs_gray):
        blurred_img = cv2.GaussianBlur(obs_gray, (3,3), 0)
        canny_edges = cv2.Canny(blurred_img, 10, 100)
        value = canny_edges.argmax(axis=-1)
        return np.max(value)

    def boss_blood_count(self, boss_blood_hsv_img):
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([360, 30, 220])
        mask = cv2.inRange(boss_blood_hsv_img, lower_white, upper_white)
        white_pixel_count = cv2.countNonZero(mask)
        return white_pixel_count

    def self_stamina_count(self, self_stamina_hsv_img):
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([360, 30, 220])
        mask = cv2.inRange(self_stamina_hsv_img, lower_white, upper_white)
        white_pixel_count = cv2.countNonZero(mask)
        return white_pixel_count

    def boss_stamina_count(self, boss_stamina_hsv_img):
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([360, 30, 220])
        mask = cv2.inRange(boss_stamina_hsv_img, lower_white, upper_white)
        white_pixel_count = cv2.countNonZero(mask)
        return white_pixel_count
    
    def take_action(self, action):
        if action == 0: #j
            directkeys.light_attack()
        elif action == 1: #m
            directkeys.left_dodge()
        elif action == 2:
            directkeys.sanlian()
        elif action == 3:
            directkeys.right_dodge()
        elif action == 4:
            directkeys.hard_attack()
        
        #elif action == 4: #r_back
           # directkeys.dodge_back()
       # elif action == 5: #hard_attack
           # directkeys.hard_attack()
        #elif action == 1: #ninja_attack
           # directkeys.ninja_attack()
        #elif action == 3: #skill_attack
            #directkeys.skill_attack()
        
    def get_reward(self, boss_blood, next_boss_blood, self_blood, next_self_blood, 
                   boss_stamina, next_boss_stamina, self_stamina, next_self_stamina, 
                   stop, emergence_break,action,boss_attack):
        print(next_self_blood,boss_blood)
        if next_self_blood < 50:     # self dead
            print("dead")
            # print("快死了，当前血量：",self_blood,"马上血量：",next_self_blood)
            if emergence_break < 2:
                reward = -6
                done = 1
                stop = 0
                emergence_break += 1
                # time.sleep(3)
                print("后跳并喝血")
                pyautogui.keyDown('S')
                directkeys.dodge()
                directkeys.dodge()
                directkeys.dodge()
                time.sleep(0.2)
                pyautogui.press('R')
                time.sleep(1)
                pyautogui.press('R')
                pyautogui.press('R')
                pyautogui.keyUp('S')
                # pyautogui.keyDown('num2')
                # pyautogui.keyDown('num2')
                # pyautogui.keyDown('num2') # 必须要3次它才能检测到，原因未知，少一次都不行
                # time.sleep(1)
                # pyautogui.keyUp('num2') # 释放按键，下一次才能正确按到
                return reward, done, stop, emergence_break
            else:
                reward = -6
                done = 1
                stop = 0
                emergence_break = 100
                # pyautogui.keyDown('num2')
                # pyautogui.keyDown('num2')
                # pyautogui.keyDown('num2') # 必须要3次它才能检测到，原因未知，少一次都不行
                # time.sleep(1)
                # pyautogui.keyUp('num2') # 释放按键，下一次才能正确按到
                print("后跳并喝血")
                pyautogui.keyDown('S')
                directkeys.dodge()
                directkeys.dodge()
                directkeys.dodge()
                time.sleep(0.2)
                pyautogui.press('R')
                time.sleep(1)
                pyautogui.press('R')
                pyautogui.press('R')
                pyautogui.keyUp('S')
                return reward, done, stop, emergence_break
        # elif next_boss_blood - boss_blood > 70 and boss_blood < 10:   #boss dead
        #     print("boss死了")
        #     if emergence_break < 2:
        #         reward = 30
        #         done = 0
        #         stop = 0
        #         emergence_break += 1
        #         return reward, done, stop, emergence_break
        #     else:
        #         reward = 30
        #         done = 0
        #         stop = 0
        #         emergence_break = 100
        #         return reward, done, stop, emergence_break
        
        else:
            reward = 0
            self_blood_reward = 0
            boss_blood_reward = 0
            self_stamina_reward = 0
            boss_stamina_reward = 0
            # if action == 0: # 二连
            #     reward  += 0
            # elif action == 1: # 闪避
            #     reward += 0
            # elif action == 2: # 三连
            #     reward += 0
            # elif action == 3: # 重棍
            #     reward -= 1;
            # print(next_self_blood - self_blood)
            # print(next_boss_blood - boss_blood)
            
            # 自己掉血扣分
            if next_self_blood - self_blood < -5:
                # if stop == 0:
                self_blood_reward = (next_self_blood - self_blood) //10
                print("掉血惩罚")
                end_defense = True
                # stop = 1
                time.sleep(0.05)
                # 防止连续取帧时一直计算掉血
            # else:
            #     stop = 0
            # 打掉boss血加分
            if next_boss_blood - boss_blood <= -18:
                print("打掉boss血而奖励")
                boss_blood_reward = (boss_blood - next_boss_blood) // 10
                boss_blood_reward = min(boss_blood_reward,10)
            # print("self_blood_reward:    ",self_blood_reward)
            # print("boss_blood_reward:    ",boss_blood_reward)
            # 成功防御加分
            if (action == 1 or action == 3) and boss_attack == True and next_self_stamina - self_stamina >= 7 and next_self_blood-self_blood == 0:
                print("完美闪避奖励")
                self_stamina_reward += 2
            elif (action == 1 or action == 3) and boss_attack == True and next_self_blood-self_blood == 0:
                print("成功闪避")
                self_stamina_reward += 0.5
            # boss架势值增加加分
            #如果什么都没做，进行惩罚
            reward = reward + self_blood_reward * 0.8 + boss_blood_reward * 1.2 + self_stamina_reward * 1.0 + boss_stamina_reward * 1.0
            done = 0
            emergence_break = 0
            # if(reward != -0.5):
            #     print("Reward of this round is:", reward)        
            return reward, done, stop, emergence_break

    def step(self, action, boss_attack):
        if(action == 0):
            print("一连")
        elif(action == 1):
            print("左闪避")
        elif(action == 2):
            print("三连")
        elif action == 3:
            print("右闪避")
        elif action == 4:
            print("重棍")
        self.take_action(action)
        
        obs_screen = grab_screen(self.obs_window)
        obs_resize = cv2.resize(obs_screen,(self.width,self.height))
        # obs_bgr = cv2.cvtColor(obs_resize,cv2.COLOR_BGR2RGB)
        # obs_rgb = cv2.cvtColor(obs_bgr,cv2.COLOR_BGR2RGB)
        # obs改为灰度
        # obs_gray = cv2.cvtColor(obs_resize, cv2.COLOR_BGR2GRAY) # 
        obs = np.array(obs_resize).reshape(-1,self.height,self.width,4)[0]
        # for test 
        # obs = obs.squeeze()
        # plt.imshow(obs, cmap='gray')
        # plt.show()
        # print(obs.shape)
        # 只狼血量统计
        sekiro_blood_img = grab_screen(self.sekiro_blood_window)
        # sekiro_blood_hsv_img = cv2.cvtColor(sekiro_blood_img, cv2.COLOR_BGR2HSV)
        sekiro_blood_gray_img = cv2.cvtColor(sekiro_blood_img, cv2.COLOR_BGR2GRAY)
        next_self_blood = self.self_blood_count(sekiro_blood_gray_img)
        # boss血量统计
        boss_blood_img = grab_screen(self.boss_blood_window)
        boss_blood_hsv_img = cv2.cvtColor(boss_blood_img, cv2.COLOR_BGR2HSV)
        next_boss_blood = self.boss_blood_count(boss_blood_hsv_img)
        # 只狼架势条统计
        sekiro_stamina_img = grab_screen(self.sekiro_stamina_window)
        sekiro_stamina_hsv_img = cv2.cvtColor(sekiro_stamina_img, cv2.COLOR_BGR2HSV)
        next_self_stamina = self.self_stamina_count(sekiro_stamina_hsv_img)
        # boss架势条统计
        boss_stamina_img = grab_screen(self.boss_stamina_window)
        boss_stamina_hsv_img = cv2.cvtColor(boss_stamina_img, cv2.COLOR_BGR2HSV)
        next_boss_stamina = self.self_stamina_count(boss_stamina_hsv_img)
        
        # print("只狼血量：",next_self_blood,"    boss血量：",next_boss_blood)
        # print("只狼架势条：",next_self_stamina,"    boss架势条：",next_boss_stamina)
        reward, done, stop, emergence_break = self.get_reward(self.boss_blood, next_boss_blood, self.self_blood, next_self_blood, 
                   self.boss_stamina, next_boss_stamina, self.self_stamina, next_self_stamina, 
                   self.stop, self.emergence_break,action,boss_attack)
        self.self_blood = next_self_blood
        self.boss_blood = next_boss_blood
        self.self_stamina = next_self_stamina
        self.boss_stamina = next_boss_stamina
        #reward += loss_reward
        return (obs, reward, done, stop, emergence_break)
        

    def pause_game(self,paused):
        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('start game')
                time.sleep(1)
            else:
                paused = True
                print('pause game')
                time.sleep(1)
        if paused:
            print('paused')
            while True:
                keys = key_check()
                # pauses game and can get annoying.
                if 'T' in keys:
                    if paused:
                        paused = False
                        print('start game')
                        time.sleep(1)
                        break
                    else:
                        paused = True
                        time.sleep(1)
        return paused

    def reset(self,initial = False):
        restart(initial)
        obs_screen = grab_screen(self.obs_window)
        obs_resize = cv2.resize(obs_screen,(self.width,self.height))
        # obs_bgr = cv2.cvtColor(obs_resize,cv2.COLOR_BGR2RGB)
        # obs_rgb = cv2.cvtColor(obs_bgr,cv2.COLOR_BGR2RGB)
        # obs_gray = cv2.cvtColor(obs_resize, cv2.COLOR_BGR2GRAY)
        obs = np.array(obs_resize).reshape(-1,self.height,self.width,4)[0]
        # # 只狼血量统计
        # sekiro_blood_img = grab_screen(self.sekiro_blood_window)
        # sekiro_blood_hsv_img = cv2.cvtColor(sekiro_blood_img, cv2.COLOR_BGR2HSV)
        # self.self_blood = self.self_blood_count(sekiro_blood_hsv_img)
        # # boss血量统计
        # boss_blood_img = grab_screen(self.boss_blood_window)
        # boss_blood_hsv_img = cv2.cvtColor(boss_blood_img, cv2.COLOR_BGR2HSV)
        # self.boss_blood = self.boss_blood_count(boss_blood_hsv_img)
        # # 只狼架势条统计
        # sekiro_stamina_img = grab_screen(self.sekiro_stamina_window)
        # sekiro_stamina_hsv_img = cv2.cvtColor(sekiro_stamina_img, cv2.COLOR_BGR2HSV)
        # self.self_stamina = self.self_stamina_count(sekiro_stamina_hsv_img)
        # # boss架势条统计
        # boss_stamina_img = grab_screen(self.boss_stamina_window)
        # boss_stamina_hsv_img = cv2.cvtColor(boss_stamina_img, cv2.COLOR_BGR2HSV)
        # self.self_stamina = self.self_stamina_count(boss_stamina_hsv_img)
        return obs

def boss_blood_count(boss_blood_hsv_img):
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([360, 30, 220])
    mask = cv2.inRange(boss_blood_hsv_img, lower_white, upper_white)
    white_pixel_count = cv2.countNonZero(mask)
    return white_pixel_count
def hsv_test(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_value = hsv_img[1,1]
    print(hsv_value)

def self_stamina_count(self_stamina_hsv_img):
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([360, 60, 225])
    mask = cv2.inRange(self_stamina_hsv_img, lower_white, upper_white)
    white_pixel_count = cv2.countNonZero(mask)
    return white_pixel_count


# def self_blood_count(sekiro_blood_hsv_img):
#     lower_white = np.array([0, 0, 180])
#     upper_white = np.array([360, 75, 225])
#     mask = cv2.inRange(sekiro_blood_hsv_img, lower_white, upper_white)
#     white_pixel_count = cv2.countNonZero(mask)
#     return white_pixel_count

# sekiro_blood_window = (138,738,243,749)  #黑神话 
# sekiro_blood_img = grab_screen(sekiro_blood_window)
# obs_gray = cv2.cvtColor(sekiro_blood_img, cv2.COLOR_BGR2GRAY)
# print(self_blood_count(obs_gray))
def self_blood_count(obs_gray):
    blurred_img = cv2.GaussianBlur(obs_gray, (3,3), 0)
    canny_edges = cv2.Canny(blurred_img, 10, 100)
    value = canny_edges.argmax(axis=-1)
    return np.max(value)

def self_power_count(self_power_hsv_img):
    lower_white = np.array([0, 0, 220])
    upper_white = np.array([360, 45, 256])
    mask = cv2.inRange(self_power_hsv_img, lower_white, upper_white)
    white_pixel_count = cv2.countNonZero(mask)
    return white_pixel_count

if __name__ == "__main__":
    env = Sekiro(observation_w=100, observation_h=200, action_dim=5)
    # while True:
    #     env.step(0)
    #     time.sleep(5)
    width = 200
    height = 175
    obs_window = (389,135,704,525) # 笔记本
    boss_blood_window = (510,692,774,701)# 黑神话
    sekiro_blood_window = (138,738,243,749)  #黑神话 
    # boss_stamina_window  = (345,78,690,81) #笔记本
    sekiro_stamina_window = (1128,725,1158,779)
    # pyautogui.keyDown('num2')
    # pyautogui.keyDown('num2')
    # pyautogui.keyDown('num2') # 必须要3次它才能检测到，原因未知，少一次都不行
    # time.sleep(1)
    # pyautogui.keyUp('num2') # 释放按键，下一次才能正确按到
    # time.sleep(0.5)
    # pyautogui.keyDown('num2')
    # pyautogui.keyDown('num2')
    # pyautogui.keyDown('num2') # 必须要3次它才能检测到，原因未知，少一次都不行
    # time.sleep(1)
    # pyautogui.keyUp('num2') # 释放按键，下一次才能正确按到
    # obs_screen = grab_screen(obs_window)
    # obs_resize = cv2.resize(obs_screen,(width,height))
    # obs = np.array(obs_resize).reshape(-1,height,width,4)[0]
    # obs = 1
    # while True:
    while True:
        self_power_window = (1194,752,1220,780)
        # self_power_window = (1209,756,1211,758)
        self_power_img = grab_screen(self_power_window)
        # hsv_test(self_power_img)
        self_power_hsv = cv2.cvtColor(self_power_img, cv2.COLOR_BGR2HSV)
        self_power = self_power_count(self_power_hsv)
        print(self_power)
    #     time.sleep(0.5)
        # sekiro_blood_img = grab_screen(sekiro_blood_window)
        # # sekiro_blood_hsv_img = cv2.cvtColor(sekiro_blood_img, cv2.COLOR_BGR2HSV)
        # obs_gray = cv2.cvtColor(sekiro_blood_img, cv2.COLOR_BGR2GRAY)
        # print(self_blood_count(obs_gray))
        # # px = sekiro_blood_hsv_img[1,20]
        # # # print(px[0],px[1],px[2])
        # # self_blood = self_blood_count(sekiro_blood_hsv_img)
        # time.sleep(0.2)
        
        # print(self_blood)
        # time.sleep(0.05)

        # sekiro_stamina_img = grab_screen(sekiro_stamina_window)
        # sekiro_stamina_hsv_img = cv2.cvtColor(sekiro_stamina_img, cv2.COLOR_BGR2HSV)
        # self_stamina = self_stamina_count(sekiro_stamina_hsv_img)
        # print(self_stamina)




                                    