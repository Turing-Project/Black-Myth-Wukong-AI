
import itertools
import sys
import torch
import os
import cv2
import utils.directkeys as directkeys
import time
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from nets.dqn_net import Q_construct
from utils.schedules import *
from replay_buffer import *
from collections import namedtuple
from nets.ResNet_boss_model import ResNet50_boss
from screen_key_grab.grabscreen import grab_screen
from torch.utils.tensorboard import SummaryWriter


OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

# 此处指定在cpu上跑
dtype = torch.FloatTensor
dlongtype = torch.LongTensor
device = "cpu"
paused = True
writer = SummaryWriter()
# # Set the logger
# logger = Logger('./logs')

# 状态对应--刀郎
index_to_label = {
    0: '冲刺砍',
    1: '旋转飞',
    2: '扔刀',
    3: '飞雷神',
    4: '锄地',
    5: '锄地起飞',
    6: '受到攻击',
    7: '普攻',
    8: '观察',
    9: '大荒星陨'
}




def dqn_learning(env,
                 optimizer_spec,
                 exploration=LinearSchedule(1000, 0.1), 
                 stopping_criterion=None,
                 replay_buffer_size=1000,
                 batch_size=32,
                 gamma=0.99,
                 learning_starts=50,
                 learning_freq=4,
                 frame_history_len=4,
                 target_update_freq=10,
                 double_dqn=False,
                 checkpoint=0):

    ################
    #  BUILD MODEL #
    ################
    paused = env.pause_game(True)


    num_actions = env.action_dim
    # 初始boss模型
    model_resnet_boss = ResNet50_boss(num_classes=10) # 刀郎或其他boss的模型
    model_resnet_boss.load_state_dict(torch.load(
        'D:/dqn_wukong/RL-ARPG-Agent-1/boss_model_baiyi.pkl'))
    model_resnet_boss.to(device)
    model_resnet_boss.eval()
    
    # 初始自身模型
    # model_resnet_malo = ResNet50_boss(num_classes=2) # 用于判断自身是否倒地
    # model_resnet_malo.load_state_dict(torch.load(
    #     'D:/dqn_wukong/RL-ARPG-Agent-1/malo_model.pkl'))
    # model_resnet_malo.to(device)
    # model_resnet_malo.eval()


    # 控制冻结和更新的参数
    for param in model_resnet_boss.parameters():
        param.requires_grad = False
        
    # Q网络初始化
    Q = Q_construct(input_dim=256, num_actions=num_actions).type(dtype)
    Q_target = Q_construct(input_dim=256, num_actions=num_actions).type(dtype)

    # load checkpoint
    if checkpoint != 0:
        checkpoint_path = "models/wukong_0825_2_1200.pth"
        Q.load_state_dict(torch.load(checkpoint_path))
        Q_target.load_state_dict(torch.load(checkpoint_path))
        print('load model success')

    # initialize optimizer
    optimizer = optimizer_spec.constructor(
        Q.parameters(), **optimizer_spec.kwargs)

    # create replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)


    ###########
    # RUN ENV #
    ###########

    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset(initial=True)
    LOG_EVERY_N_STEPS = 10
    SAVE_MODEL_EVERY_N_STEPS = 100
    episode_rewards = []
    episode_reward = 0
    episode_cnt = 0
    loss_fn = nn.MSELoss() # 损失函数
    loss_cnt = 0
    reward_cnt = 0 # 统计reward次数，到一定次数统计一次reward
    reward_10 = 0  # 用来画reward曲线
    boss_attack = False # 表征boss是否处于攻击状态
    initial_steal = True # 第一次上来先偷一棍
    for t in itertools.count(start=checkpoint):
        # t += 5500
        # Check stopping criterion 可自定义
        if stopping_criterion is not None and stopping_criterion(env, t):
            break
        # Step the env and store the transition
        # store last frame, return idx used later
        last_stored_frame_idx = replay_buffer.store_frame(last_obs)
        # get observatitrons to input to Q network (need to append prev frames)
        observations = replay_buffer.encode_recent_observation()
        # print(observations.shape)
        if initial_steal:
            print("偷一棍")
            directkeys.hard_attack_long() # 黑神话特色偷一刀
            initial_steal = False
        obs = torch.from_numpy(observations).unsqueeze(
            0).type(dtype)  
        obs = obs[:, :3, 20:180, 5:165]  
        
        # # 用于查看截取的图像是否正确
        # obs = obs.flip(dims = (1,))
        # array2 = obs.squeeze() 
        # array2 = array2.permute(1,2,0) 
        # array2 = array2.cpu().numpy()
        # array2 = array2 - array2.min()
        # array2 = array2 / array2.max()
        # plt.imshow(array2)
        # plt.colorbar()
        # plt.show()
        
        output_boss, intermediate_results_boss = model_resnet_boss(obs)
        max_values_boss, indices_boss = torch.max(output_boss, dim=1)
        print("当前帧判断的boss状态:", index_to_label[indices_boss.item()])
        if indices_boss.item() != 6 and indices_boss.item() != 8:
            boss_attack = True
        else:
            boss_attack = False
            
        # before learning starts, choose actions randomly
        if t < learning_starts:
            action = np.random.randint(num_actions)
        else:
            # epsilon greedy exploration
            sample = random.random()
            threshold = exploration.value(t)
            if sample > threshold:
                q_value_all_actions = Q(intermediate_results_boss)
                q_value_all_actions = q_value_all_actions.cpu()
                action = ((q_value_all_actions).data.max(1)[1])[0]
            else:
                action = torch.IntTensor(
                    [[np.random.randint(num_actions)]])[0][0]
                

        '''---------------自身状态提取--------------'''
        
        self_power_window = (1566,971,1599,1008) # 棍势点
        self_power_img = grab_screen(self_power_window)
        self_power_hsv = cv2.cvtColor(self_power_img, cv2.COLOR_BGR2HSV)
        self_power = env.self_power_count(self_power_hsv)  # >50一段 >100第二段
        
        self_endurance_window = (186,987,311,995) # 耐力条
        self_endurance_img = grab_screen(self_endurance_window)
        endurance_gray = cv2.cvtColor(self_endurance_img,cv2.COLOR_BGR2GRAY)
        self_endurance = env.self_endurance_count(endurance_gray) # 为0是满或空 中间是准确的
        
        ding_shen_window = (1458,851,1459,852)
        ding_shen_img = grab_screen(ding_shen_window)
        hsv_img = cv2.cvtColor(ding_shen_img, cv2.COLOR_BGR2HSV)
        hsv_value = hsv_img[0,0]
        
        ding_shen_available = False
        if hsv_value[2] >= 130:
            ding_shen_available = True
        
        self_window = (548,770,1100,1035) # 倒地位置
        self_img = grab_screen(self_window)
        screen_reshape = cv2.resize(self_img,(175,200))[20:180,5:165,:3]
        screen_reshape = screen_reshape.transpose(2,0,1)
        screen_reshape = screen_reshape.reshape(1,3,160,160)
        tensor_malo = torch.from_numpy(screen_reshape).type(dtype = torch.float32)

        '''-----------------------------------------------'''
        
        '''--------------------手动约束部分-----------------'''
        selected_num = random.choice([1, 3]) # 1,3分别是左翻滚和右翻滚
        # if indices_boss.item() == 4 or indices_boss.item() == 9:  # 锄地
        #     action = torch.tensor([selected_num])
        # elif indices_boss.item() == 5 or indices_boss.item() == 1 or indices_boss.item() == 2:  # 锄地起飞
        #     action = torch.tensor([selected_num])
        # elif indices_boss.item() == 7:  # 普攻
        #     action = torch.tensor([selected_num])
        # elif indices_boss.item() == 6:  # 受到攻击
        #     action = torch.tensor([2])
        # elif indices_boss.item() == 3: # 飞雷神
        #     action = torch.tensor([selected_num])
        # elif indices_boss.item() == 0:  # 冲刺砍或扔刀
        #     if self_power > 100:
        #         action = torch.tensor([4])
        # elif indices_boss.item() == 8:  # 观察
        #     if self_power > 100:
        #         action = torch.tensor([4])
        #     else:
        #         action = torch.tensor([0])
        if action != 3 and action != 1 and self_endurance < 30 and self_endurance != 0: # 攻击但是没有耐力了
            action = torch.tensor([5])
        # if indices_boss.item() == 1 and self_power > 50: # 可识破
        #     action = torch.tensor([7])
        # for state in state_list:
        #     if state != 6 and state != 8:
        #         action = torch.tensor([selected_num])
        if ding_shen_available == True:
            action = torch.tensor([6])
            
        # # 额外判断是否倒地，倒地则必须翻滚
        # res,embed = model_resnet_malo(tensor_malo)
        # max_values_boss, indices_self = torch.max(res, dim=1)
        # if indices_self.item() == 0: # 猴倒地
        #     print("倒地了，翻滚")
        #     action = torch.tensor([selected_num])
        
        '''----------------约束结束----------------------'''
        # state_list.append(indices_boss.item()) # 维护boss状态
        # print(state_list)
        obs, reward, done, stop, emergence_break = env.step(
            action, boss_attack)
        if action == 4:  # 把重棍处理成三连棍
            action = 2
        elif action == 5: # 把歇脚回气力处理成轻棍
            action = 0
        elif action == 6: # 定身处理成重棍
            action = 2
        elif action == 7: # 识破处理成重棍
            action = 2
        if reward_cnt % 30 == 0:
            reward_10 += reward
            writer.add_scalars(
                "reward", {"reward_10":  reward_10}, (reward_cnt) / 30)
            reward_10 = 0
            reward_cnt += 1
        else:
            reward_10 += reward
            reward_cnt += 1
        episode_reward += reward
        replay_buffer.store_effect(last_stored_frame_idx, action, reward, done)
        if done:
            obs = env.reset()
            episode_rewards.append(episode_reward)
            writer.add_scalar("reward_episode", episode_reward, episode_cnt)
            episode_cnt += 1
            print("current episode reward %d" % episode_reward)
            episode_reward = 0
        last_obs = obs
        env.pause_game(False)
        # Perform experience replay and train the network
        # if the replay buffer contains enough samples..
        last_time = time.time()
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            last_time = time.time()
            # sample transition batch from replay memory
            # done_mask = 1 if next state is end of episode
            obs_t, act_t, rew_t, obs_tp1, done_mask = replay_buffer.sample(
                batch_size)      
            obs_t = torch.tensor(obs_t, dtype=torch.float32)
            obs_t = obs_t[:, :3, 20:180, 5:165]
            obs_t = obs_t.to(device)
            act_t = torch.tensor(act_t, dtype=torch.long).to(device)
            rew_t = torch.tensor(rew_t, dtype=torch.float32).to(device)
            obs_tp1 = torch.tensor(obs_tp1, dtype=torch.float32)
            obs_tp1 = obs_tp1[:, :3, 20:180, 5:165]
            obs_tp1 = obs_tp1.to(device)
            done_mask = torch.tensor(done_mask, dtype=torch.float32).to(device)
            # input batches to networks
            # get the Q values for current observations (Q(s,a, theta_i))
            output_boss_, intermediate_results_boss = model_resnet_boss(obs_t)
            output_boss_, intermediate_results_boss_tp1 = model_resnet_boss(
                obs_tp1)
            q_values = Q(intermediate_results_boss)
            q_s_a = q_values.gather(1, act_t.unsqueeze(1))
            q_s_a = q_s_a.squeeze()
            if (double_dqn):
                # ------------
                # double DQN
                # ------------
                # get Q values for best actions in obs_tp1
                # based off the current Q network
                # max(Q(s',a',theta_i)) wrt a'
                q_tp1_values = Q(intermediate_results_boss_tp1)
                q_tp1_values = q_tp1_values.detach()
                _, a_prime = q_tp1_values.max(1)
                # get Q values from frozen network for next state and chosen action
                # Q(s', argmax(Q(s',a',theta_i), theta_i_frozen)) (argmax wrt a')
                q_target_tp1_values = Q_target(intermediate_results_boss_tp1)
                q_target_tp1_values = q_target_tp1_values.detach()
                q_target_s_a_prime = q_target_tp1_values.gather(
                    1, a_prime.unsqueeze(1))
                q_target_s_a_prime = q_target_s_a_prime.squeeze()
                # if current state is end of episode, then there is no next Q value
                q_target_s_a_prime = (1 - done_mask) * q_target_s_a_prime
                expected_q = rew_t + gamma * q_target_s_a_prime
            else:
                # -------------
                # regular DQN
                # -------------
                # get Q values for best actions in obs_tp1
                # based off frozen Q network
                # max(Q(s',a',theta_i_frozen)) wrt a'
                q_tp1_values = Q_target(intermediate_results_boss_tp1)
                q_tp1_values = q_tp1_values.detach()
                q_s_a_prime, a_prime = q_tp1_values.max(1)
                # if current state is end of episode, then there is no next Q value
                q_s_a_prime = (1 - done_mask) * q_s_a_prime
                # Compute Bellman error
                # r + gamma * Q(s', a', theta_i_frozen) - Q(s, a, theta_i)
                expected_q = rew_t + gamma * q_s_a_prime
            time_before_optimization = time.time()
            # 计算loss
            loss = loss_fn(expected_q, q_s_a)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar("loss_dqn", loss.item(), loss_cnt)
            loss_cnt += 1
            num_param_updates += 1
            print('optimization took {} seconds'.format(
                time.time()-time_before_optimization))
            # update target Q network weights with current Q network weights
            if num_param_updates % target_update_freq == 0:
                Q_target.load_state_dict(Q.state_dict())
            print('loop took {} seconds'.format(time.time()-last_time))
            env.pause_game(False)
        # 4. Log progress
        if t % SAVE_MODEL_EVERY_N_STEPS == 0:
            if not os.path.exists("models"):
                os.makedirs("models")
            if not os.path.exists("models_res"):
                os.makedirs("models_res")
            model_save_path = "models/wukong_0904_1_%d.pth" % (t)
            torch.save(Q.state_dict(), model_save_path)
        # episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-10:])
            best_mean_episode_reward = max(
                best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0:
            print("---------------------------------")
            print("Timestep %d" % (t,))
            print("learning started? %d" % (t > learning_starts))
            print("mean reward (10 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.kwargs['lr'])
            sys.stdout.flush()