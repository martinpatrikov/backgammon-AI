import argparse
import pickle
import warnings

import numpy as np
import torch
import gym
import utils

# suppress warnings from gym / torch
warnings.filterwarnings('ignore')

# CONSTANTS
MODEL_PATH    = './model.pth'
STATE_FILE    = 'state.pickle'
AGENT_FILE    = 'agent_colour.pickle'
ENV_ID        = 'gym_backgammon:backgammon-v0'

PLAYER_NAMES  = {0: 'Red', 1: 'Black'}
PIECE_SYMBOLS = {0: 'X', 1: 'O'}
OPP_MAP       = {'X': 0, 'O': 1}


def parse_args():
    p = argparse.ArgumentParser(description="Play Backgammon")
    p.add_argument('-begin',
                   action='store_true',
                   help='Start a new game')
    p.add_argument('-starting_player',
                   dest='colour',
                   choices=['X', 'O'],
                   default='X',
                   help='Who moves first (X or O)')
    p.add_argument('-ai',
                   nargs=2,
                   type=int,
                   metavar=('DIE1', 'DIE2'),
                   help='Let the AI play on a given roll')
    p.add_argument('-human',
                   nargs='+',
                   help='Human move(s) as pairs of points or "bar"')
    p.add_argument('-skip',
                   action='store_true',
                   help='Skip turn if no move possible')
    return p.parse_args()


# STATE PERSISTENCE
def save_state(state, agent_colour, only_agent=False):
    if not only_agent:
        with open(STATE_FILE, 'wb') as f:
            pickle.dump(state, f)
    with open(AGENT_FILE, 'wb') as f:
        pickle.dump(agent_colour, f)


def restore_env(env):
    try:
        with open(STATE_FILE, 'rb') as f:
            state = pickle.load(f)
        with open(AGENT_FILE, 'rb') as f:
            agent_colour = pickle.load(f)
    except FileNotFoundError:
        print("No saved game found. Use `-begin` to start a new one.")
        exit(1)

    # restore on both wrapped & unwrapped env
    env.game.restore_state(state)
    env.current_agent = agent_colour
    env.unwrapped.current_agent = agent_colour
    return state, agent_colour


# GAME ACTIONS
def start_game(env, args):
    # keep resetting until chosen colour starts
    while True:
        agent_colour, roll, state = env.reset()
        if agent_colour == OPP_MAP[args.colour]:
            break

    env.render()
    print(f"\n{PIECE_SYMBOLS[agent_colour]} player to start.")
    save_state(env.game.save_state(), agent_colour)


def ai_move(env, args, model):
    env.reset()  # just initialize game object
    state, agent_colour = restore_env(env)
    agent = utils.TD_BG_Agent(env, agent_colour, model)

    raw = args.ai
    # handle doubles: four dice on a double roll
    if raw[0] == raw[1]:
        dice = [raw[0]] * 4
    else:
        dice = list(raw)

    roll = np.array(dice)
    signed = -roll if agent_colour == 1 else roll

    # valid = env.get_valid_actions(tuple(signed))
    agent_roll = tuple(-signed)
    valid = env.get_valid_actions(agent_roll)
    if not valid:
        print("\nNo legal moves for that roll.\n")
        env.render()
        next_agent = env.get_opponent_agent()
        save_state(env.game.save_state(), next_agent)
        return

    # action = agent.best_action(tuple(-signed))
    action = agent.best_action(agent_roll)
    next_state, _, terminal, _ = env.step(action)

    print()
    if terminal:
        print("🎉 AI has won!")
    else:
        prob = agent.network(torch.Tensor(next_state))[0].item()
        print(f"AI win‐prob estimate: {prob:.2f}")

    print(f"\nAI chose action: {action}\n")
    env.render()

    next_agent = env.get_opponent_agent()
    print(f"{PIECE_SYMBOLS[next_agent]} player to move.")
    save_state(env.game.save_state(), next_agent)


def human_move(env, args):
    env.reset()
    state, agent_colour = restore_env(env)

    # parse human tokens into integer/bar list
    moves = []
    for tok in args.human:
        if tok.lower() == 'bar':
            moves.append('bar')
        else:
            moves.append(int(tok))
    actions = tuple(map(tuple, np.array(moves, dtype=object).reshape(-1, 2)))

    try:
        next_state, _, terminal, _ = env.step(actions)
    except AssertionError:
        raise ValueError(f"Invalid move {actions} for player {PIECE_SYMBOLS[agent_colour]}")

    print()
    env.render()
    if terminal:
        print(f"{PIECE_SYMBOLS[agent_colour]} player won!")
    else:
        next_agent = env.get_opponent_agent()
        print(f"{PIECE_SYMBOLS[next_agent]} player to move.")
        save_state(env.game.save_state(), next_agent)


# ENTRY POINT
def main():
    args = parse_args()
    env   = gym.make(ENV_ID, disable_env_checker=True)
    model = torch.load(MODEL_PATH, weights_only=False)

    if args.begin:
        start_game(env, args)
    elif args.ai is not None:
        ai_move(env, args, model)
    elif args.skip:
        env.reset()
        state, agent_colour = restore_env(env)
        print(f"\n{PIECE_SYMBOLS[agent_colour]} player skipped.\n")
        env.render()
        next_agent = env.get_opponent_agent()
        print(f"{PIECE_SYMBOLS[next_agent]} player to move.")
        save_state(state, next_agent, only_agent=True)
    else:
        human_move(env, args)


if __name__ == '__main__':
    main()
