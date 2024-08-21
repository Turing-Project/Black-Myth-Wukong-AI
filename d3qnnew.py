#from cv2 import getOptimalNewCameraMatrix, threshold
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
from torch.autograd import Variable
import sys
import os
import gym.spaces
import itertools
import numpy as np
import random
from collections import namedtuple
from replay_buffer import *
from schedules import *
from resnet18 import res18
from dqn_net3 import Q_construct
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from ResNet_player_model import ResNet50_player
# from efficientnet import *
# from utils.gym_setup import *
# from logger import Logger
import time
from torch.utils.tensorboard import SummaryWriter


OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

# CUDA variables
# USE_CUDA = torch.cuda.is_available()
# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
# dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 此处指定在cpu上跑
dtype = torch.FloatTensor
dlongtype = torch.LongTensor
device = "cpu"
paused = True
writer = SummaryWriter()
# # Set the logger
# logger = Logger('./logs')
def to_np(x):
    return x.data.cpu().numpy() 
# 状态对应
index_to_label = {
    0:'下砍',
    1:'突刺危',
    2:'横砍',
    3:'翻滚下劈',
    4:'跳劈',
    5:'飞渡',
    6:'射箭',
    7:'下段危',
    8:'冲刺扫',
    9:'其他' 
}
def dqn_learning(env,
          env_id,
          q_func,
          optimizer_spec,
          exploration=LinearSchedule(1000, 0.1), #此处和buffer的100原先是1000
          stopping_criterion=None,
          replay_buffer_size=100,
          batch_size=32,
          gamma=0.99,
          learning_starts=50,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10,
          double_dqn=False,
          dueling_dqn=False,
          checkpoint = 0):


    ################
    #  BUILD MODEL #
    ################
    paused = env.pause_game(True)

    img_w = env.width
    img_h = env.height
    img_c = 1
    input_shape = (img_h, img_w, frame_history_len * img_c)
    in_channels = input_shape[2]

    num_actions = env.action_dim

    model_resnet_boss = ResNet50_player(num_classes=10)
    # # 加载已训练好的模型
    model_resnet_boss.load_state_dict(torch.load('D:/d3qn_sekiro/RL-ARPG-Agent-3/player_model.pkl'))
    model_resnet_boss.to(device)
    model_resnet_boss.eval()
    # model_resnet_sekiro = torch.load('resnet_model_sekiro_round11_0.pth')
    # model_resnet_sekiro.to(device)
    # model_resnet_sekiro.eval()
    
    # criterion = nn.NLLLoss()
    # optimizer_boss = optim.Adam(model_resnet_boss.parameters(), lr=0.00005)
    # optimizer_sekiro = optim.Adam(model_resnet_sekiro.parameters(), lr = 0.00005)

    # 控制冻结和更新的参数
    for param in model_resnet_boss.parameters():
        param.requires_grad = False
    # for param in model_resnet_sekiro.parameters():
    #     param.requires_grad = False
    # for param in model_resnet_boss.conv5_x.parameters():
    #     param.requires_grad = True
    # for param in model_resnet_boss.avg_pool.parameters():
    #     param.requires_grad = True
    # for param in model_resnet_boss.fc1.parameters():
    #     param.requires_grad = True
    # for param in model_resnet_boss.fc2.parameters():
    #     param.requires_grad = True
    # for param in model_resnet_boss.fc3.parameters():
    #     param.requires_grad = True
    # for param in model_resnet_sekiro.conv5_x.parameters():
    #     param.requires_grad = True
    # for param in model_resnet_sekiro.avg_pool.parameters():
    #     param.requires_grad = True
    # for param in model_resnet_sekiro.fc1.parameters():
    #     param.requires_grad = True
    # for param in model_resnet_sekiro.fc2.parameters():
    #     param.requires_grad = True
    # for param in model_resnet_sekiro.fc3.parameters():
    #     param.requires_grad = True
    # 新的Q和Qtarget网络定义
    # Q = q_func(in_channels = in_channels, num_actions = num_actions).type(dtype)
    # Q_target = q_func(in_channels = in_channels, num_actions = num_actions).type(dtype)
    Q = Q_construct(input_dim = 256,num_actions = num_actions).type(dtype)
    Q_target = Q_construct(input_dim = 256,num_actions = num_actions).type(dtype)
    # Q = efficientnet(num_classes=num_actions, net="B0", pretrained=False).type(dtype)
    # Q_target = efficientnet(num_classes=num_actions, net="B0", pretrained=False).type(dtype)

    # load checkpoint
    if checkpoint != 0:
        add_str = ''
        if (double_dqn):
            add_str = 'double' 
        if (dueling_dqn):
            add_str = 'dueling'
        checkpoint_path = "models/real_ulti4_0810_2400.pth"
        Q.load_state_dict(torch.load(checkpoint_path))
        Q_target.load_state_dict(torch.load(checkpoint_path))
        print('load model success')

    # initialize optimizer
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # create replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ########

    ###########
    # RUN ENV #
    ###########

    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    # print("初始化而reset")
    last_obs = env.reset(initial = True)
    LOG_EVERY_N_STEPS = 10
    SAVE_MODEL_EVERY_N_STEPS = 100
    episode_rewards = []
    episode_reward = 0
    # print("正式进入循环")
    cnt_loss_epoch = 0
    episode_cnt = 0
    loss_fn = nn.MSELoss()
    loss_cnt = 0
    reward_cnt = 0 
    reward_10 = 0
    loss_cnt_res = 0
    # 统计连续防御次数，连续防御达到3次认为进入防御状态
    continuous_defense = 0
    is_defending = False
    end_defense = False
    for t in itertools.count(start=checkpoint):
        # t += 5500
        ### Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### Step the env and store the transition
        # store last frame, return idx used later
        last_stored_frame_idx = replay_buffer.store_frame(last_obs)

        # get observatitrons to input to Q network (need to append prev frames)
        observations = replay_buffer.encode_recent_observation()
        #print(observations.shape)

        # before learning starts, choose actions randomly
        if t < learning_starts:
            action = np.random.randint(num_actions)
        else:
            # epsilon greedy exploration
            sample = random.random()
            threshold = exploration.value(t)
            if sample > threshold:
                obs = torch.from_numpy(observations).unsqueeze(0).type(dtype) # 1,4,175,200
                obs = obs[:,:3,20:180,5:165]# 1,3,128,128
                # obs = obs.flip(dims = (1,))# 1,3,128,128
                # array2 = obs.squeeze() # 3,128,128
                # array2 = array2.permute(1,2,0) # 128,128,3
                # array2 = array2.cpu().numpy()
                # array2 = array2 - array2.min()
                # array2 = array2 / array2.max()
                # plt.imshow(array2)
                # plt.colorbar()
                # plt.show()
                output_boss,intermediate_results_boss = model_resnet_boss(obs)
                max_values_boss, indices_boss = torch.max(output_boss, dim=1)
                print("boss状态:",index_to_label[indices_boss.item()])
                if indices_boss.item() == 1: # 突刺
                    action = torch.tensor([2])
                # elif indices_boss.item() == 4: # 跳劈
                #     action = torch.tensor([1])
                # elif indices_boss.item() == 5: # 飞渡
                #     action = torch.tensor([9])
                # elif indices_boss.item() == 7: # 下段危
                #     action = torch.tensor([3])
                else:
                    q_value_all_actions= Q(intermediate_results_boss)
                    q_value_all_actions = q_value_all_actions.cpu()
                    action = ((q_value_all_actions).data.max(1)[1])[0]
                # 进入防御状态则一直防御，直到成功防御或掉血为止
                # if action == 1:
                #     continuous_defense += 1
                #     if continuous_defense >= 4:
                #        is_defending = True
                #     if continuous_defense >= 8:
                #         end_defense = True
                # if end_defense == True:
                #     is_defending = False
                #     continuous_defense = 0
                # elif action != 1:
                #     continuous_defense = 0

            else:
                action = torch.IntTensor([[np.random.randint(num_actions)]])[0][0]
        # print("即将和环境互动")
        obs, reward, done, stop, emergence_break, end_defense = env.step(action,is_defending)
        if action == 9: # 把防飞渡处理成防御
            action = 1
        if action == 3: # 把防下段危的跳处理成防御
            action = 1
        if reward_cnt % 30 == 0:
            reward_10 += reward
            writer.add_scalars("reward",{"reward_10":  reward_10},(reward_cnt) / 30)        
            reward_10 = 0
            reward_cnt += 1
        else:
            reward_10 += reward
            reward_cnt += 1
        # # for test
        # obs = obs.squeeze()
        # plt.imshow(obs, cmap='gray')
        # plt.show()
        # obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).type(dtype)
        # output_boss,intermediate_results_boss = model_resnet_boss(obs[:,:,:120,:])
        # output_sekiro,intermediate_results_sekiro = model_resnet_sekiro(obs[:,:,120:,:])
        # print(output_boss,output_sekiro)
        # time.sleep(10)
        episode_reward += reward
        # print("即将存入buffer")
        # store effect of action 
        replay_buffer.store_effect(last_stored_frame_idx, action, reward, done)

        # reset env if reached episode boundary
        if done:
            obs = env.reset()
            episode_rewards.append(episode_reward)
            writer.add_scalar("reward_episode",episode_reward,episode_cnt)
            episode_cnt += 1
            # print("以上由于done而reset")
            print("current episode reward %d" % episode_reward)
            episode_reward = 0

        # update last_obs
        last_obs = obs
        env.pause_game(False)
        ### Perform experience replay and train the network
        # if the replay buffer contains enough samples..
        last_time = time.time()
        if (learning_starts != 1086 and t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            last_time = time.time()
            # sample transition batch from replay memory
            # done_mask = 1 if next state is end of episode
            # print("即将从Buffer中取出")
            obs_t, act_t, rew_t, obs_tp1, done_mask = replay_buffer.sample(batch_size)      # 2,4,175,200
            obs_t = torch.tensor(obs_t, dtype=torch.float32)
            obs_t = obs_t[:,:3,20:180,5:165]
            # obs_t = obs_t.flip(dims = (1,))
            obs_t = obs_t.to(device)
            act_t = torch.tensor(act_t, dtype=torch.long).to(device)
            rew_t = torch.tensor(rew_t, dtype=torch.float32).to(device)
            obs_tp1 = torch.tensor(obs_tp1, dtype=torch.float32)
            obs_tp1 = obs_tp1[:,:3,20:180,5:165]
            # obs_tp1 = obs_tp1.flip(dims = (1,))
            obs_tp1 = obs_tp1.to(device)
            done_mask = torch.tensor(done_mask, dtype=torch.float32).to(device)
            # print("取出完成")
            # input batches to networks
            # get the Q values for current observations (Q(s,a, theta_i))
            # print('Buffer took {} seconds'.format(time.time()-last_time))
            
            time_resnet_start = time.time()
            output_boss_,intermediate_results_boss = model_resnet_boss(obs_t)
            # obs = obs[:,:3,:128,30:158]# 1,3,128,128t
            # obs = obs.flip(dims = (1,))# 1,3,128,128
            # # array2 = obs.squeeze() # 3,128,128
            # # array2 = array2.permute(1,2,0) # 128,128,3
            # # array2 = array2.cpu().numpy()
            # # array2 = array2 - array2.min()
            # # array2 = array2 / array2.max()
            # # plt.imshow(array2)
            # # plt.colorbar()
            # # plt.show()
            output_boss_,intermediate_results_boss_tp1 = model_resnet_boss(obs_tp1)
            # print('Resnet took {} seconds'.format(time.time()-time_resnet_start))
            # print(obs_q_t.shape)
            time_before_Q = time.time()
            q_values = Q(intermediate_results_boss)
            q_s_a = q_values.gather(1, act_t.unsqueeze(1))
            q_s_a = q_s_a.squeeze()
            # print('Getting q value took {} seconds'.format(time.time()-time_before_Q))
            time_before_Q_calculation = time.time()
            if (double_dqn):

                #------------
                # double DQN
                #------------

                # get Q values for best actions in obs_tp1
                # based off the current Q network
                # max(Q(s',a',theta_i)) wrt a'
                q_tp1_values= Q(intermediate_results_boss_tp1)
                q_tp1_values = q_tp1_values.detach()
                _, a_prime = q_tp1_values.max(1)

                # get Q values from frozen network for next state and chosen action
                # Q(s', argmax(Q(s',a',theta_i), theta_i_frozen)) (argmax wrt a')
                q_target_tp1_values= Q_target(intermediate_results_boss_tp1)
                q_target_tp1_values = q_target_tp1_values.detach()
                q_target_s_a_prime = q_target_tp1_values.gather(1, a_prime.unsqueeze(1))
                q_target_s_a_prime = q_target_s_a_prime.squeeze()
                # if current state is end of episode, then there is no next Q value
                q_target_s_a_prime = (1 - done_mask) * q_target_s_a_prime

                expected_q = rew_t + gamma * q_target_s_a_prime
            
            else:
                #-------------
                # regular DQN
                #-------------

                # get Q values for best actions in obs_tp1
                # based off frozen Q network
                # max(Q(s',a',theta_i_frozen)) wrt a'
                q_tp1_values= Q_target(intermediate_results_boss_tp1)
                q_tp1_values = q_tp1_values.detach()
                q_s_a_prime, a_prime = q_tp1_values.max(1)

                # if current state is end of episode, then there is no next Q value
                q_s_a_prime = (1 - done_mask) * q_s_a_prime

                # Compute Bellman error
                # r + gamma * Q(s', a', theta_i_frozen) - Q(s, a, theta_i)
                expected_q = rew_t + gamma * q_s_a_prime
            
            # print('DQN calculation took {} seconds'.format(time.time()-time_before_Q_calculation))
            
            time_before_optimization = time.time()
            # ------------更新resnet 和dqn网络 ----------------------------------------------------------------
            # 计算loss
            loss_boss_graph = 0
            loss_sekiro_graph = 0
            # if t > 1000:
            #     loss_boss = criterion(torch.log(output_boss), act_t)  
            #     loss_sekiro = criterion(torch.log(output_sekiro),act_t)
            #     loss_boss_graph = loss_boss.item()
            #     loss_sekiro_graph = loss_sekiro.item()
            loss = loss_fn(expected_q, q_s_a)
            # zerograd
            # if t > 1000:
            #     optimizer_boss.zero_grad()
            #     optimizer_sekiro.zero_grad()
            optimizer.zero_grad()
            # loss反向传播
            # if t > 1000:
            #     loss_boss.backward(retain_graph=True)
            #     loss_sekiro.backward(retain_graph=True)
            loss.backward()
            # 优化器执行
            # if t > 1000:
            #     optimizer_boss.step()
            #     optimizer_sekiro.step()            
            optimizer.step()
            # 添加 loss 到 TensorBoard
            writer.add_scalar("loss_dqn",loss.item(),loss_cnt)
            # if t > 1000:
            #     writer.add_scalar("loss_resnet_boss",loss_boss_graph,loss_cnt)
            #     writer.add_scalar("loss_resnet_sekiro",loss_sekiro_graph, loss_cnt)
            loss_cnt += 1
            #---------------------------------更新完成------------------------------------------------
            num_param_updates += 1

            print('optimization took {} seconds'.format(time.time()-time_before_optimization))
            # update target Q network weights with current Q network weights
            if num_param_updates % target_update_freq == 0:
                Q_target.load_state_dict(Q.state_dict())

            # (2) Log values and gradients of the parameters (histogram)
            #if t % LOG_EVERY_N_STEPS == 0:
                #for tag, value in Q.named_parameters():
                   # tag = tag.replace('.', '/')
                    #logger.histo_summary(tag, to_np(value), t+1)
                    #logger.histo_summary(tag+'/grad', to_np(value.grad), t+1)
            #####
            print('loop took {} seconds'.format(time.time()-last_time))
            env.pause_game(False)

        ## 4. Log progress
        if t % SAVE_MODEL_EVERY_N_STEPS == 0:
            if not os.path.exists("models"):
                os.makedirs("models")
            if not os.path.exists("models_res"):
                os.makedirs("models_res")
            add_str = ''
            if (double_dqn):
                add_str = 'double' 
            if (dueling_dqn):
                add_str = 'dueling'
            model_save_path = "models/real_ulti4_0815_%d.pth" %(t)
            torch.save(Q.state_dict(), model_save_path)
            # model_resnet_boss_path = "models_res/ulti_boss_%s_%s_%d.pth" %(str(env_id), add_str, t)
            # torch.save(model_resnet_boss.state_dict(), model_resnet_boss_path)
            # model_resnet_sekiro_path = "models_res/ulti_sekiro_%s_%s_%d.pth" %(str(env_id), add_str, t)
            # torch.save(model_resnet_sekiro.state_dict(), model_resnet_sekiro_path)

        #episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-10:])
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
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







