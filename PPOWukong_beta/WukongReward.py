import cv2
import numpy as np
import time
import pytesseract 

from matplotlib import pyplot as plt
class WukongReward:
    '''Reward Class'''


    '''Constructor'''
    def __init__(self, config):
        pytesseract.pytesseract.tesseract_cmd = config["PYTESSERACT_PATH"]        #Setting the path to pytesseract.exe
        self.GAME_MODE = config["GAME_MODE"]
        self.DEBUG_MODE = config["DEBUG_MODE"]
        self.max_hp = config["PLAYER_HP"]                             #This is the hp value of your character. We need this to capture the right length of the hp bar.
        self.prev_hp = 1.0     
        self.curr_hp = 1.0
        self.time_since_dmg_taken = time.time()
        self.death = False
        self.max_stam = config["PLAYER_STAMINA"]                     
        self.curr_stam = 1.0
        self.curr_boss_hp = 1.0
        self.time_since_boss_dmg = time.time() 
        self.time_since_pvp_damaged = time.time()
        self.time_alive = time.time()
        self.boss_death = False    
        self.game_won = False    
        self.image_detection_tolerance = 0.02               #The image detection of the hp bar is not perfect. So we ignore changes smaller than this value. (0.02 = 2%)


    '''Detecting the current player hp'''
    def get_current_hp(self, frame):
        x = 200
        y=979
        hp_image=frame[y:y+17,x:x+154]
        image_gray = cv2.cvtColor(hp_image, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(image_gray, (3, 3), 0)
        canny_edges = cv2.Canny(blurred_img, 10, 100)
        value = canny_edges.argmax(axis=-1)
        curr_hp=np.max(value)/154



        return curr_hp

    '''Detecting the current player stamina'''
    def get_current_stamina(self, frame):
        x = 200
        y = 1014
        hp_image = frame[y:y + 8, x:x + 154]
        image_gray = cv2.cvtColor(hp_image, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(image_gray, (3, 3), 0)
        canny_edges = cv2.Canny(blurred_img, 10, 100)
        value = canny_edges.argmax(axis=-1)
        self.curr_stam=np.max(value)/141
        return self.curr_stam
    

    '''Detecting the current boss hp'''
    def get_boss_hp(self, frame):
        x = 755
        y = 911
        # 截取指定范围的图像部分
        image = frame[y:y + 10,x:x + 400]
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(image_gray, (3, 3), 0)
        canny_edges = cv2.Canny(blurred_img, 10, 100)
        value = canny_edges.argmax(axis=-1)
        self.boss_hp=np.max(value)/398
        return self.boss_hp
    

    '''Detecting if the boss is damaged in PvE'''           #🚧 This is not implemented yet!!
    def detect_boss_damaged(self, frame):
        cut_frame = frame[863:876, 462:1462]
        
        lower = np.array([23,210,0])                                            #This filter really inst perfect but its good enough bcause stamina is not that important
        upper = np.array([25,255,255])                                           #Also Filter
        hsv = cv2.cvtColor(cut_frame, cv2.COLOR_RGB2HSV)                       #Apply the filter
        mask = cv2.inRange(hsv, lower, upper)                                   #Also apply
        matches = np.argwhere(mask==255)                                        #Number for all the white pixels in the mask


        if len(matches) > 30:                                                   #if there are more than 30 white pixels in the mask, return true
            return True
        else:
            return False

    

    '''Detecting if the enemy is damaged in PvP'''
    def detect_pvp_damaged(self, frame):
        cut_frame = frame[150:400, 350:1700]
        
        lower = np.array([24,210,0])                                            #This filter really inst perfect but its good enough bcause stamina is not that important
        upper = np.array([25,255,255])                                           #Also Filter
        hsv = cv2.cvtColor(cut_frame, cv2.COLOR_RGB2HSV)                       #Apply the filter
        mask = cv2.inRange(hsv, lower, upper)                                   #Also apply
        matches = np.argwhere(mask==255)                                        #Number for all the white pixels in the mask
        if len(matches) > 30:                                                   #if there are more than 30 white pixels in the mask, return true
            return True
        else:
            return False
    

    '''Detecting if the duel is won in PvP'''
    def detect_win(self, frame):
        cut_frame = frame[730:800, 550:1350]
        lower = np.array([0,0,75])                  #Removing color from the image
        upper = np.array([255,255,255])
        hsv = cv2.cvtColor(cut_frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        pytesseract_output = pytesseract.image_to_string(mask,  lang='eng',config='--psm 6 --oem 3') #reading text from the image cutout
        game_won = "Combat ends in your victory!" in pytesseract_output or "combat ends in your victory!" in pytesseract_output             #Boolean if we see "combat ends in your victory!" on the screen
        return game_won
    

    '''Debug function to render the frame'''
    def render_frame(self, frame):
        cv2.imshow('debug-render', frame)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

 
    '''Update function that gets called every step and returns the total reward and if the agent died or the boss died'''
    def update(self, frame, first_step):
        #📍 1 Getting current values
        #📍 2 Hp Rewards
        #📍 3 Boss Rewards
        #📍 4 PvP Rewards
        #📍 5 Total Reward / Return


        '''📍1 Getting/Setting current values'''
        self.curr_hp = self.get_current_hp(frame)                   
        self.curr_stam = self.get_current_stamina(frame)            
        self.curr_boss_hp = self.get_boss_hp(frame)
        if first_step: self.time_since_dmg_taken = time.time() - 10 #Setting the time_since_dmg_taken to 10 seconds ago so we dont get a negative reward at the start of the game           

        self.death = False
        if self.curr_hp <= 0.01 + self.image_detection_tolerance:   #If our hp is below 1% we are dead
            self.death = True
            self.curr_hp = 0.0

        self.boss_death = False
        if self.GAME_MODE == "PVE":                                 #Only if we are in PVE mode
            if self.curr_boss_hp <= 0.01:                           #If the boss hp is below 1% the boss is dead (small tolerance because we want to be sure the boss is actually dead)
                self.boss_death = True

        
        '''📍2 Hp Rewards'''
        hp_reward = 0
        if not self.death:                           
            if self.curr_hp > self.prev_hp + self.image_detection_tolerance:        #Reward if we healed)
                hp_reward = 100                  
            elif self.curr_hp < self.prev_hp - self.image_detection_tolerance:      #Negative reward if we took damage
                hp_reward = -69
                self.time_since_dmg_taken = time.time()
        else:
            hp_reward = -420                                                        #Large negative reward for dying

        time_since_taken_dmg_reward = 0                                  
        if time.time() - self.time_since_dmg_taken > 5:                             #Reward if we have not taken damage for 5 seconds (every step for as long as we dont take damage)
            time_since_taken_dmg_reward = 15


        self.prev_hp = self.curr_hp     #Update prev_hp to curr_hp


        '''📍3 Boss Rewards'''
        if self.GAME_MODE == "PVE":                                             #Only if we are in PVE mode
            boss_dmg_reward = 0
            if self.boss_death:                                                     #Large reward if the boss is dead
                boss_dmg_reward = 420
            else:
                if self.detect_boss_damaged(frame):  #Reward if we damaged the boss (small tolerance because its a large bar)
                    boss_dmg_reward = 69
                    self.time_since_boss_dmg = time.time()
                if time.time() - self.time_since_boss_dmg > 5:                      #Negative reward if we have not damaged the boss for 5 seconds (every step for as long as we dont damage the boss)
                    boss_dmg_reward = -25                                               
            

            percent_through_fight_reward = 0
            if self.curr_boss_hp < 0.97:                                            #Increasing reward for every step we are alive depending on how low the boss hp is
                percent_through_fight_reward = self.curr_boss_hp * 15            


        '''📍4 PVP rewards'''
        pvp_reward = 0
        if self.GAME_MODE != "PVE":                                              #Only if we are in PVP mode
            enemy_damaged = self.detect_pvp_damaged(frame)                          #Detect if the enemy is damaged
            if enemy_damaged:                                                       #Reward if the enemy is damaged
                pvp_reward = 69                         
                self.time_since_pvp_damaged = time.time()
            else:
                if time.time() - self.time_since_pvp_damaged > 5:                   #Negative reward if we have not damaged the enemy for 5 seconds (every step for as long as we dont damage the enemy)
                    pvp_reward = -25
                    #print("🔫 Duelist not damaged for 5s")
                else:
                    pvp_reward = 0

            #staying alive reward
            '''                     #a time alive reward could cause problems because the agent will still get rewarded even if performing bad when the time alive reward is higher than the other punishments
            time_alive_reward = 0
            if time.time() - self.time_alive > 5:                                   #Reward if we have been alive for 5 seconds (we give an increasinig reward for every second we are alive)
                time_alive_reward = time.time() - self.time_alive - 5
                print("🕒 Time alive reward: ", time_alive_reward)
                pvp_reward += time_alive_reward
            '''

            #winning
            self.game_won = self.detect_win(frame)    #not implemented yet
            if self.game_won:
                pvp_reward = 420


        '''📍5 Total Reward / Return'''
        if self.GAME_MODE == "PVE":                                                 #Only if we are in PVE mode
            total_reward = hp_reward + boss_dmg_reward + time_since_taken_dmg_reward + percent_through_fight_reward
        else:
            total_reward = hp_reward + time_since_taken_dmg_reward + pvp_reward
        
        total_reward = round(total_reward, 3)

        return total_reward, self.death, self.boss_death, self.game_won

    


'''Testing code'''
if __name__ == "__main__":
    env_config = {
        "PYTESSERACT_PATH": r'C:\Program Files\Tesseract-OCR\tesseract.exe',    # Set the path to PyTesseract
        "MONITOR": 1,           #Set the monitor to use (1,2,3)
        "DEBUG_MODE": False,    #Renders the AI vision (pretty scuffed)
        "GAME_MODE": "PVE",     #PVP or PVE
        "BOSS": 8,              #1-6 for PVE (look at walkToBoss.py for boss names) | Is ignored for GAME_MODE PVP
        "BOSS_HAS_SECOND_PHASE": True,  #Set to True if the boss has a second phase (only for PVE)
        "PLAYER_HP": 1679,      #Set the player hp (used for hp bar detection)
        "PLAYER_STAMINA": 121,  #Set the player stamina (used for stamina bar detection)
        "DESIRED_FPS": 24       #Set the desired fps (used for actions per second) (24 = 2.4 actions per second) #not implemented yet       #My CPU (i9-13900k) can run the training at about 2.4SPS (steps per secons)
    }
    reward = WukongReward(env_config)

    IMG_WIDTH = 1920                                #Game capture resolution
    IMG_HEIGHT = 1080

    import mss
    sct = mss.mss()
    monitor = sct.monitors[1]

    sct_img = sct.grab(monitor)
    frame = cv2.cvtColor(np.asarray(sct_img), cv2.COLOR_BGRA2RGB)
    frame = frame[46:IMG_HEIGHT + 46, 12:IMG_WIDTH + 12]    #cut the frame to the size of the game
    print(reward.get_current_stamina(frame))

    # reward.update(frame, True)

    # time.sleep(1)
    # reward.update(frame, False)

        