# -*- coding: utf-8 -*-
# File: play.py

import argparse
import random
from typing import Optional

from agent import Agent
from game import Symbol, TicTacToe


def play(
    size: int = 3,
    pkl_path: Optional[str] = None,
    agent_1: bool = False,
    agent_2: bool = False,
    seed: Optional[int] = None,
) -> None:
    # Check arguments
    if pkl_path is None and (agent_1 or agent_2):
        raise ValueError(f"`pkl_path` cannot be None when either `agent_1` or `agent_2` is True.")

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
        if step == 0 and agent_1:
            action = random.choice(game.action_space)
            game(*action, player)
        elif (step % 2 == 0 and agent_1) or (step % 2 == 1 and agent_2):
            for action, _ in agent(game.state):
                if game(*action, player):
                    break
        else:
            choice = input(f"{player.value}: ")
            action = (int(choice[1:]) - 1, ord(choice[0]) - ord("A"))
            while not game(*action, player):
                choice = input(f"{player.value}: ")
                action = (int(choice[1:]) - 1, ord(choice[0]) - ord("A"))

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
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    play(size=args.size, pkl_path=args.pkl_path, agent_1=args.agent_1, agent_2=args.agent_2, seed=args.seed)
