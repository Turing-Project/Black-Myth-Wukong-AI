import torch.optim as optim
import argparse
from dqn import dqn_learning, OptimizerSpec
from schedules import LinearSchedule
from env_wukong import Wukong

BATCH_SIZE = 2
REPLAY_BUFFER_SIZE = 1000 
FRAME_HISTORY_LEN = 1
TARGET_UPDATE_FREQ = 100
GAMMA = 0.96
LEARNING_FREQ = 4
LEARNING_RATE = 0.001
ALPHA = 0.90
EPS = 0.0005
EXPLORATION_SCHEDULE = LinearSchedule(800, 0.05)
LEARNING_STARTS = 0 
CHECKPOINT = 0

def Wukong_learn(env, double_dqn,checkpoint):

    optimizer = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS)
    )
    dqn_learning(
        env=env,
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
        checkpoint = checkpoint
        )



def main():
    parser = argparse.ArgumentParser(description='RL agents for Wukong')
    parser.add_argument("--gpu", type=int, default=0, help="ID of GPU to be used")
    parser.add_argument("--double-dqn", type=int, default=1, help="double dqn - 0 = No, 1 = Yes")
    parser.add_argument("--checkpoint", type=int, default=0, help="checkpoint")

    args = parser.parse_args()
    


    # Run training
    double_dqn = (args.double_dqn == 1)
    env = Wukong(observation_w=175, observation_h=200, action_dim=4)
    Wukong_learn(env,double_dqn=double_dqn, checkpoint=0)

if __name__ == '__main__':
    main()