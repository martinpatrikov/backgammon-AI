import numpy as np
import gym
import time
from itertools import count
import random
from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS, TOKEN
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Network(nn.Module):
    """Simple feedforward network estimating win-probability for White."""
    def __init__(self):
        super(Network, self).__init__()
        # 198-dimensional input → 50 hidden units → 1 output
        self.x_layer = nn.Sequential(
            nn.Linear(198, 50),
            nn.Sigmoid()
        )
        self.y_layer = nn.Sequential(
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def forward(self, state):
        # Pass state through network to get win probability
        xh = self.x_layer(state)
        win_probability = self.y_layer(xh)
        return win_probability


class TD_BG_Agent:
    """Temporal-Difference learning agent for backgammon."""
    def __init__(self, env, player, network):
        self.env = env
        self.player = player               # WHITE or BLACK
        self.network = network             # evaluation network
        self.elig_traces = None            # eligibility traces for weights
        self.lamda = 0.7                   # trace decay
        self.learning_rate = 0.1           # step size
        self.gamma = 0.99                  # discount factor

    def roll_dice(self):
        """Roll two dice; use negative values for White to match env convention."""
        if self.player == WHITE:
            return (-random.randint(1,6), -random.randint(1,6))
        else:
            return ( random.randint(1,6),  random.randint(1,6))

    def best_action(self, roll):
        """Evaluate all legal actions for `roll` and choose the best one."""
        actions = list(self.env.get_valid_actions(roll))
        values = []
        current_state = self.env.game.save_state()  # snapshot current board

        for a in actions:
            # simulate action
            next_state, reward, terminal, winner = self.env.step(a)
            # evaluate resulting state
            val = self.network(torch.Tensor(next_state))
            values.append(val)
            # restore board to before action
            self.env.game.restore_state(current_state)

        # extract scores and select argmax for White, argmin for Black
        scores = [v.detach().cpu().item() for v in values]
        if self.player == WHITE:
            idx = int(np.argmax(scores))
        else:
            idx = int(np.argmin(scores))
        return actions[idx]

    def update(self, current_prob, future_prob, reward):
        """
        Perform TD(λ) update on network weights, using
        δ = r + γ·V(next) - V(current).
        """
        # Zero gradients and backprop current_prob to fill .grad
        self.network.zero_grad()
        current_prob.backward()

        delta_t = reward + self.gamma * future_prob - current_prob
        # Update weights manually using eligibility traces
        with torch.no_grad():
            for i, weights in enumerate(self.network.parameters()):
                # update eligibility trace
                self.elig_traces[i] = (
                    self.gamma * self.lamda * self.elig_traces[i]
                    + weights.grad
                )
                # apply weight update
                new_w = weights + self.learning_rate * delta_t * self.elig_traces[i]
                weights.copy_(new_w)
        return delta_t


def train(agent_white, agent_black, env, n_episodes=1, max_time_steps=3000):
    """
    Train two TD agents (White vs Black) by self-play.
    Returns a list of TD errors (losses) per completed game.
    """
    losses = []
    agents = {WHITE: agent_white, BLACK: agent_black}
    wins   = {WHITE: 0, BLACK: 0}

    for episode in range(n_episodes):
        # reset env, pick starting player and roll
        agent_colour, roll, state = env.reset()
        agent = agents[agent_colour]

        # reset eligibility traces for both agents
        for ag in (agent_white, agent_black):
            ag.elig_traces = [torch.zeros(p.shape, requires_grad=False)
                              for p in ag.network.parameters()]

        # play up to max_time_steps moves
        for t in range(max_time_steps):
            # on first move, use initial roll; afterwards roll dice
            if t > 0:
                roll = agent.roll_dice()

            current_prob = agent.network(torch.Tensor(state))
            valid_actions = env.get_valid_actions(roll)

            if not valid_actions:
                # no move: switch player and continue
                agent_colour = env.get_opponent_agent()
                agent = agents[agent_colour]
                continue

            # choose and apply best action
            best_act = agent.best_action(roll)
            next_state, reward, terminal, winner = env.step(best_act)
            future_prob = agent.network(torch.Tensor(next_state))

            # terminal update
            if terminal and winner is not None:
                # reward sign normalization for Black win
                if winner == BLACK:
                    reward = -1
                loss = agent.update(current_prob, future_prob, reward)
                losses.append(loss / (t+1))
                wins[winner] += 1

                # print progress
                total = wins[WHITE] + wins[BLACK]
                w_pct = 100 * wins[WHITE] / total
                print(f"Episode {episode}: winner={winner}, W%={w_pct:.2f}")
                break

            # non-terminal update and switch player
            agent.update(current_prob, future_prob, reward)
            agent_colour = env.get_opponent_agent()
            agent = agents[agent_colour]
            state = next_state

    return losses
