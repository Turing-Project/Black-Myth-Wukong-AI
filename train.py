import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import torch.optim as optim
import argparse

# from efficientnet import efficientnet
from d3qnnew import dqn_learning, OptimizerSpec
#from utils.atari_wrappers import *
#from utils.gym_setup import *
from schedules import *
from env_sekiro import Sekiro



# Global Variables
# Extended data table 1 of nature paper
BATCH_SIZE = 2
REPLAY_BUFFER_SIZE = 1000 # 原来1000
FRAME_HISTORY_LEN = 1
TARGET_UPDATE_FREQ = 100
GAMMA = 0.96
LEARNING_FREQ = 4
LEARNING_RATE = 0.001
ALPHA = 0.90
EPS = 0.0005
EXPLORATION_SCHEDULE = LinearSchedule(800, 0.05)
LEARNING_STARTS = 1086 # 1086 一个奇怪的数字让训练时跳过更新过程，不训练只输出
CHECKPOINT = 0

# torch.cuda.set_device(0)

def sekiro_learn(env, env_id, double_dqn, dueling_dqn, checkpoint):

    # def stopping_criterion(env, t):
    #     # notice that here t is the number of steps of the wrapped env,
    #     return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps
    
    optimizer = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS)
    )
    dqn_learning(
        env=env,
        env_id=env_id,
        optimizer_spec=optimizer,
        exploration=EXPLORATION_SCHEDULE,
        stopping_criterion=None,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGET_UPDATE_FREQ,
        double_dqn=double_dqn,
        dueling_dqn=dueling_dqn,
        checkpoint = checkpoint
        )
    #env.close()



def main():
    parser = argparse.ArgumentParser(description='RL agents for sekiro')
    #subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    #train_parser = subparsers.add_parser("train", help="train an RL agent for sekiro games")
    parser.add_argument("--gpu", type=int, default=0, help="ID of GPU to be used")
    parser.add_argument("--double-dqn", type=int, default=1, help="double dqn - 0 = No, 1 = Yes")
    parser.add_argument("--dueling-dqn", type=int, default=0, help="dueling dqn - 0 = No, 1 = Yes")
    parser.add_argument("--checkpoint", type=int, default=0, help="checkpoint")

    args = parser.parse_args()
    

    # command
    # if (args.gpu != None):
    #     if torch.cuda.is_available():
    #         torch.cuda.set_device(args.gpukvjrvjmvjkvjm)
    #         print("CUDA Device: %d" %torch.cuda.current_device())

    # Run training
    double_dqn = (args.double_dqn == 1)
    dueling_dqn = (args.dueling_dqn == 1)
    checkpoint = args.checkpoint
    env = Sekiro(observation_w=175, observation_h=200, action_dim=4)
    print("double_dqn %d, dueling_dqn %d" %(double_dqn, dueling_dqn))
    sekiro_learn(env, 4, double_dqn=double_dqn, dueling_dqn=dueling_dqn, checkpoint=1) #此处load模型

if __name__ == '__main__':
    main()