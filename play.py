# -*- coding: utf-8 -*-
# File: play.py

import argparse
import random
import sys
from typing import Optional

from agent import Agent
from game import Symbol, TicTacToe


def play(
    size: int = 3,
    pkl_path: Optional[str] = None,
    agent_1: bool = False,
    agent_2: bool = False,
    random_prob: float = 0.0,
    seed: Optional[int] = None,
) -> None:
    def get_user_action(player: Symbol) -> tuple[int, int]:
        while True:
            choice = input(f"{player.value}: ")
            try:
                row, col = int(choice[1:]) - 1, ord(choice[0].upper()) - ord("A")
                assert 0 <= row <= 25 or 0 <= col <= 25
                return row, col
            except KeyboardInterrupt:
                sys.exit()
            except:
                continue

    # Check arguments
    if pkl_path is None and (agent_1 or agent_2):
        raise ValueError(f"`pkl_path` cannot be None when either `agent_1` or `agent_2` is True.")
    if not (0 <= random_prob <= 1):
        raise ValueError(f"`random_prob` is expected to be in range [0, 1], got {random_prob}.")

    # Seed random number generator
    random.seed(seed)

    # Initialize
    game = TicTacToe(size=size)
    if agent_1 or agent_2:
        agent = Agent()
        agent.load(pkl_path)
    game.print_board(game.board)
    print("\n")

    player = Symbol.P1
    step = 0
    while True:
        if (step % 2 == 0 and agent_1) or (step % 2 == 1 and agent_2):
            if (step == 0 and agent_1) or random.random() <= random_prob:
                action = random.choice(game.action_space)
                while not game(*action, player):
                    action = random.choice(game.action_space)
            else:
                for action, _ in agent(game.state):
                    if game(*action, player):
                        break
        else:
            action = get_user_action(player)
            while not game(*action, player):
                action = get_user_action(player)

        # Show information
        print(f"{player.value} chose {chr(ord('A') + action[1])}{action[0] + 1}\n->")
        game.print_board(game.board)
        print("\n")

        # Check termination
        if (winner := game.has_winner()) is not None or game.is_full():
            print("Draw!" if winner is None else f"The winner is {winner}!")
            break

        # Proceed to the next player
        player = Symbol.P2 if player == Symbol.P1 else Symbol.P1
        step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", type=int, default=3)
    parser.add_argument("-p", "--pkl_path", type=str, default=None)
    parser.add_argument("-a1", "--agent_1", action="store_true")
    parser.add_argument("-a2", "--agent_2", action="store_true")
    parser.add_argument("-r", "--random_prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    play(
        size=args.size,
        pkl_path=args.pkl_path,
        agent_1=args.agent_1,
        agent_2=args.agent_2,
        random_prob=args.random_prob,
        seed=args.seed,
    )
