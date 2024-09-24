# AI-Wukong: RL-based ARPG GameBot
A simple Reinforcement-Learning AI based on DQN/PPO and CV model
<br>
一个简单的基于强化学习与计算机视觉模型的预训练动作游戏人工智能框架，仅限交流与科普。


![image](https://img.shields.io/badge/License-Apache--2.0-green) ![image](https://img.shields.io/badge/License-MIT-orange)  ![image](https://img.shields.io/badge/License-Anti--996-red)  ![image](https://img.shields.io/badge/pypi-v0.0.1a4-yellowgreen) ![image](https://img.shields.io/badge/stars-%3C%201k-blue) ![image](https://img.shields.io/badge/issues-1%20open-brightgreen)



## 项目简介
AI-Wukong是基于机器学习、强化学习与计算机视觉等技术所构建的自主学习式AI框架，目前第一版模型针对《黑神话悟空》的场景化任务，对比传统的纯DQN模型或纯GPT-4o多模态方案，在真实测试中AI的性能有明显提高。
| 项目作者        | 主页1           | 主页2  |
| ------------- |:------:|:----:|
| 图灵的猫       | [Bilibili](https://space.bilibili.com/371846699) |[Github](https://github.com/Y1ran) |

#### 开发人员名单：
- 陈鹤飞（上海交通大学）
- 陈泽宇（东京工业大学）
- 蓝魔digital（北京邮电大学）


## 框架说明
- [x] 基于DQN/PPO+预训练框架，AI可以自主进行战斗场景学习
- [x] 战斗核心是小规模参数量的强化学习以及ResNet等识别模型
- [x] 框架模块化设计，支持替换其他模型或DQN相关变体
- [x] 跑图与战斗解耦，通过多模态大模型进行探索交互，战斗前切换至RL


## 本地环境
* Ubuntu 18.04.2/ Windows10 x86
* Python >= 3.10
* Tensorflow-gpu 1.15.2
* Pytorch 2.3.1
* opencv_python
* CUDA >= 11.8.0
* CuDNN >= 7.6.0
* Steam/Wegame


## 系统结构
整个框架分为2个模块，每个模块之间解耦，可单独迭代或替换。
它有六组解耦合的根模块，分别是捕捉画面的识、预测出招的算、负责交互的触、用于跑图的探、记录数据的聚、以及最核心的斗战。


整体框架：
![image](https://github.com/user-attachments/assets/9fdf5cc4-7e09-4a96-be21-dcee108a3879)

### 战斗核心
#### 视觉
我们的方法是通过每Nms一次的“眨眼”频率来获取屏幕上的帧画面（N>1）resize后实时分割，截取敌人的RGB图像喂给视觉网络，以获得当前帧的状态。
通过它，AI就能计算敌人的位置、姿态和可能的攻击方式并作出应对。主要使用的方法有：
- 三层CNN+全连接
- 预训练ResNet
- Openpose姿态识别（测试）

![image](https://github.com/user-attachments/assets/1f2e3c6a-29e5-460c-ad5f-86c66585232f)

#### 决策
Reinforcement Learning——强化学习。简单来说，就是让智能体学习在不断变化的环境中，通过采取不同行动来最大化收益。
所谓收益，其实就是环境对智能体的动作给出的奖励或惩罚。比如使boss掉血或者成功闪避，AI会获得些微奖励，触发闪身或使BOSS硬直，则获得大量奖励。而自己掉血则施加一个惩罚等等。根据这些反馈来更新模型，AI就能学会如何更好的出招。
![image](https://github.com/user-attachments/assets/bbcde20a-67d4-4c78-b4a6-a08fd6583602)
主要使用的两套AI算法，其核心都是优化以上目标函数

### 游戏分辨率设置

游戏分辨率设置为1680x1050,并将比例调整为16：9，将游戏窗口化置于屏幕左上角，使得左边框左边缘和上边框上边缘紧贴显示器边缘。(适用于2560x1600显示器)

若显示器非2560x1600，需要修改env_wukong.py中的boss_blood_window等的数值以及dqn.py中的self_power_window等的数值。

### 游戏键位设置

将J设置为轻攻击，M设置为重攻击，O为加速跑，K为闪避。法术、喝药与连击等复杂操作没有固定建议，可以自行设定。

### 其他

1. [PPOWukong_beta](https://github.com/Turing-Project/RL-ARPG-Agent/tree/main/PPOWukong_beta "PPOWukong_beta")为PPO实现，其余为DQN。[pre_train](https://github.com/Turing-Project/RL-ARPG-Agent/tree/main/pre_train "pre_train")中文件用于预训练resnet。
2. 大模型目前无法给出合理的加点，因此需要在战斗模块启动前为agent配置好灵光点搭配
3. 战斗模组的反应时间可以做到0.2s，比大模型要快很多倍。如果想要更快，可以尝试dropout或者换模型
4. 视觉模块对于非人形怪的识别效果很差。一个思路是尝试新的pose识别框架或者自行训练

 ![image](https://github.com/user-attachments/assets/f270af3e-2187-4fff-8d04-7d98fc2f19aa)


## demo
可直接参考视频：https://www.bilibili.com/video/BV1qE421c7mU

跑图模块需要配置`.env`环境文件，其中需要填写对应的LLM-KEY（用哪个就填哪个）:
```
OA_OPENAI_KEY = "abc123abc123abc123abc123abc123ab"
RF_CLAUDE_AK = "abc123abc123abc123abc123abc123ab" # Access Key for Claude
RF_CLAUDE_SK = "123abc123abc123abc123abc123abc12" # Secret Access Key for Claude
IDE_NAME = "Code"
```
我们用的是GPT-4o，Key从这里获取[OpenAI](https://platform.openai.com/api-keys).


### 安装依赖

采用了目前python API调用所需的主流依赖库，用如下命令安装或确认相关依赖：

```bash
# Clone the repository
git clone https://github.com/Turing-Project/RL-ARPG-Agent
cd RL-ARPG-Agent

# Create a new conda environment
conda create --name RL-ARPG-dev python=3.10
conda activate RL-ARPG-dev
pip install -r requirements.txt
```

## 开发日志

* 2024.06.23 本地项目建立
* 2024.06.30 RL模型架构
* 2024.07.15 只狼demo-弦一狼
* 2024.07.24 法环demo-大树守卫
* 2024.08.01 视觉网络优化+预训练
* 2024.08.12 LLM-Based跑图测试
* 2024.08.20 黑神话线上测试
* 2024.08.30 通关广智
* 2024.09.24 代码开源发布


## Citation
```
@misc{AntiFraudChatBot,
  author = {Turing's Cat},
  title = {AI-Wukong: RL-based ARPG GameBot},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Turing-Project/RL-ARPG-Agent}},
}
```


## 参考资料
[1] CRADLE: Empowering Foundation Agents Towards General Computer Control, BAAI, 2024.<br>
[2] Playing Atari with Deep Reinforcement Learning，V. Mnih et al., NIPS Workshop, 2013.<br>
[3] Human-level control through deep reinforcement learning, V. Mnih et al., Nature, 2015. <br>
[4] Dueling Network Architectures for Deep Reinforcement Learning. Z. Wang et al., arXiv, 2015. <br>
[5] Deep Reinforcement Learning with Double Q-learning, H. van Hasselt et al., arXiv, 2015.<br>


## 免责声明
该项目中的内容仅供技术研究与科普，不作为任何结论性依据，不提供任何商业化应用授权
