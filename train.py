import numpy as np
import gym
import time
from itertools import count
import random
from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS, TOKEN  # game constants
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models  # for potential pretrained networks
import utils
import argparse

# set up command-line arguments for number of episodes and model checkpoint
parser = argparse.ArgumentParser(description="Training our model")
parser.add_argument('-e', '--episodes',
                    dest='n_episodes',
                    type=int,
                    default=3,
                    help='Number of training episodes')
parser.add_argument('-m', '--model',
                    dest='model_checkpoint',
                    type=str,
                    default='old',
                    help='Name of existing model file (without .pth) or "new" for fresh start')
args = parser.parse_args()

# initialize the backgammon environment
env = gym.make('gym_backgammon:backgammon-v0')

# print chosen checkpoint identifier
print(f"Using model checkpoint: {args.model_checkpoint}")

def main():
    # load an existing model or initialize a new network
    if args.model_checkpoint != 'new':
        print('Loading existing model checkpoint...')
        # Torch load returns the saved network instance
        network = torch.load(f"./{args.model_checkpoint}.pth")
        print(f"Loaded model: {args.model_checkpoint}")
    else:
        print('Initializing a new network from scratch...')
        network = utils.Network()  # fresh network architecture
        # zero all parameters to start training afresh
        for p in network.parameters():
            nn.init.zeros_(p)

    # create temporal-difference agents for White and Black using the shared network
    agent_white = utils.TD_BG_Agent(env, WHITE, network)
    agent_black = utils.TD_BG_Agent(env, BLACK, network)

    # run self-play training for specified number of episodes
    losses = utils.train(
        agent_white,
        agent_black,
        env,
        n_episodes=args.n_episodes
    )

    # after training, save the updated network parameters to a new checkpoint
    timestamp = int(time.time())
    checkpoint_name = f"{args.model_checkpoint}_{timestamp}.pth"
    torch.save(agent_white.network, f"./{checkpoint_name}")
    print(f"Saved trained model to {checkpoint_name}")

    # plot training losses over episodes
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel('Episode')
    ax.set_ylabel('TD Error')
    ax.set_title('Training Loss over Episodes')

    # save and display the loss curve
    plt.savefig('losses.png')
    plt.show()


if __name__ == '__main__':
    main()
