import numpy as np
import torch
from grabscreen import grab_screen
import cv2
import time
import directkeys as directkeys
from getkeys import key_check
from restart import restart
import matplotlib.pyplot as plt

class Sekiro(object):
    def __init__(self, observation_w, observation_h, action_dim):
        super().__init__()

        self.observation_dim = observation_w * observation_h
        self.width = observation_w
        self.height = observation_h

        self.action_dim = action_dim

        # self.obs_window = (379,160,675,500) # 显示器
        self.obs_window =  (354,180,695,521) # 笔记本
        
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
        self.sekiro_blood_window = (138,737,327,752)  #黑神话 
        # self.stamina_window = (351,66,683,530) #显示器
        # self.blood_window = (103,113,605,848)  # 显示器 1600*900
        self.boss_stamina_window = (345,78,690,81) #笔记本
        # self.sekiro_stamina_window = (426,542,626,545) #笔记本
        self.sekiro_stamina_window = (1128,725,1158,779)
        # self.stamina_window = (549,86,1074,88)  # 显示器1600*900`

        self.boss_blood = 0
        self.self_blood = 0
        self.boss_stamina = 0
        self.self_stamina = 0

        self.stop = 0
        self.emergence_break = 0

    def self_blood_count(self, sekiro_blood_hsv_img):
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([360, 30, 220])
        mask = cv2.inRange(sekiro_blood_hsv_img, lower_white, upper_white)
        white_pixel_count = cv2.countNonZero(mask)
        return white_pixel_count

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
            directkeys.attack()
        elif action == 1: #m
            directkeys.defense()
        elif action == 3: #k
            directkeys.dodge_back()
        elif action == 2: #r
            directkeys.dodge_forward()
        elif action == 4: #stay still
            directkeys.skill_attack()
        elif action == 5:
            directkeys.ninja_attack()
        elif action == 9:
            directkeys.feidu()
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
                   stop, emergence_break,action):
        end_defense = False
        if next_self_blood < 3:     # self dead
            end_defense = True
            # print("快死了，当前血量：",self_blood,"马上血量：",next_self_blood)
            if emergence_break < 2:
                reward = -4
                done = 1
                stop = 0
                emergence_break += 1
                return reward, done, stop, emergence_break,end_defense
            else:
                reward = -4
                done = 1
                stop = 0
                emergence_break = 100
                return reward, done, stop, emergence_break,end_defense
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
            if action == 1:
                reward  += 0
            elif action == 0:
                reward += 0
            elif action == 2:
                reward += 0
            elif action == 3:
                reward += 0;
            else:
                reward = -1
            # print(next_self_blood - self_blood)
            # print(next_boss_blood - boss_blood)
            
            # 自己掉血扣分
            if next_self_blood - self_blood < -5:
                # if stop == 0:
                self_blood_reward = -5
                print("掉血惩罚")
                end_defense = True
                # stop = 1
                time.sleep(0.05)
                # 防止连续取帧时一直计算掉血
            # else:
            #     stop = 0
            reward_flag = False
            # 打掉boss血加分
            if next_boss_blood - boss_blood <= -10:
                print("打掉boss血而奖励")
                reward_flag = True
                boss_blood_reward = 3.3
            # print("self_blood_reward:    ",self_blood_reward)
            # print("boss_blood_reward:    ",boss_blood_reward)
            # 成功防御加分
            if action == 1 and next_self_stamina - self_stamina >= 10 and next_self_blood-self_blood == 0:
                print("成功防御奖励")
                reward_flag =True
                end_defense = True
                self_stamina_reward = 1.0
            # boss架势值增加加分
            if next_boss_stamina - boss_stamina >= 30:
                reward_flag = True
                if action == 1 and next_boss_stamina - boss_stamina >= 30:
                    boss_stamina_reward = 1.5
                    end_defense = True
                    print("弹刀奖励！")
                elif next_boss_stamina - boss_stamina >= 30:
                    print("攻击增加架势奖励")
                    boss_stamina_reward = 1.0
                else:
                    boss_stamina_reward = 0
            #如果什么都没做，进行惩罚
            if next_boss_stamina - boss_stamina <= -10 or (next_boss_stamina == 0 and boss_stamina == 0) and reward_flag == False:
                print("boss架势值减少或无架势值惩罚")
                boss_stamina_reward = -2.0
            reward = reward + self_blood_reward * 0.8 + boss_blood_reward * 1.2 + self_stamina_reward * 1.0 + boss_stamina_reward * 1.0
            done = 0
            emergence_break = 0
            # if(reward != -0.5):
            #     print("Reward of this round is:", reward)        
            return reward, done, stop, emergence_break,end_defense

    def step(self, action, is_defending):
        if is_defending == True:
            print("保持防御")
            self.take_action(1)
        else:
            if(action == 0):
                print("进攻")
            elif(action == 3):
                print("后闪")
            elif(action == 1):
                print("防御")
            elif(action == 2):
                print("闪避")
            elif(action == 4):
                print("技能")
            elif(action == 5):
                print("忍义手")
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
        sekiro_blood_hsv_img = cv2.cvtColor(sekiro_blood_img, cv2.COLOR_BGR2HSV)
        next_self_blood = self.self_blood_count(sekiro_blood_hsv_img)
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
        reward, done, stop, emergence_break,end_defense = self.get_reward(self.boss_blood, next_boss_blood, self.self_blood, next_self_blood, 
                   self.boss_stamina, next_boss_stamina, self.self_stamina, next_self_stamina, 
                   self.stop, self.emergence_break,action)
        self.self_blood = next_self_blood
        self.boss_blood = next_boss_blood
        self.self_stamina = next_self_stamina
        self.boss_stamina = next_boss_stamina
        #reward += loss_reward
        return (obs, reward, done, stop, emergence_break,end_defense)
        

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
    upper_white = np.array([360, 30, 220])
    mask = cv2.inRange(self_stamina_hsv_img, lower_white, upper_white)
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
    sekiro_blood_window = (138,737,327,752)  #黑神话 
    # boss_stamina_window  = (345,78,690,81) #笔记本
    sekiro_stamina_window = (1128,725,1158,779)
    # obs_screen = grab_screen(obs_window)
    # obs_resize = cv2.resize(obs_screen,(width,height))
    # obs = np.array(obs_resize).reshape(-1,height,width,4)[0]
    # obs = 1
    while True:
        # boss_blood_img = grab_screen(boss_blood_window)
        # boss_blood_hsv_img = cv2.cvtColor(boss_blood_img, cv2.COLOR_BGR2HSV)
        # boss_blood = boss_blood_count(boss_blood_hsv_img)
        # print(boss_blood)
        # time.sleep(0.5)
        
        sekiro_stamina_img = grab_screen(sekiro_stamina_window)
        sekiro_stamina_hsv_img = cv2.cvtColor(sekiro_stamina_img, cv2.COLOR_BGR2HSV)
        self_stamina = self_stamina_count(sekiro_stamina_hsv_img)
        print(self_stamina)




                                    