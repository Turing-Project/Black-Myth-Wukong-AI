# RL-ARPG-Agent

AI demo for playing ARPG/Soul-like game with RL frame

## 游戏分辨率设置

游戏分辨率设置为1680x1050,并将比例调整为16：9，将游戏窗口化置于屏幕左上角，使得左边框左边缘和上边框上边缘紧贴显示器边缘。(适用于2560x1600显示器)

若显示器非2560x1600，需要修改env_wukong.py中的boss_blood_window等的数值以及dqn.py中的self_power_window等的数值。

## 游戏键位设置

将J设置为轻攻击，M设置为重攻击，O为加速跑，K为闪避。

## 其他

[PPOWukong_beta](https://github.com/Turing-Project/RL-ARPG-Agent/tree/main/PPOWukong_beta "PPOWukong_beta")为PPO实现，其余为DQN。[pre_train](https://github.com/Turing-Project/RL-ARPG-Agent/tree/main/pre_train "pre_train")中文件用于预训练resnet。
