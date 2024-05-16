# -*- coding: utf-8 -*-
# File: test.py

import argparse
import random
from typing import Optional

from tqdm import tqdm

from agent import Agent
from game import Symbol, TicTacToe


def test(
    pkl_path: str,
    size: int = 3,
    epochs: int = 100000,
    seed: Optional[int] = None,
) -> None:
    # Seed random number generator
    random.seed(seed)

    # Initialize
    game = TicTacToe(size=size)
    agent = Agent()
    agent.load(pkl_path)

    for idx, desc in enumerate(("Agent vs. Random", "Random vs. Agent", "Agent vs. Agent")):
        result = {Symbol.P1.value: 0, Symbol.P2.value: 0, None: 0}
        for _ in tqdm(range(epochs), desc=desc):
            game.clear()
            player = Symbol.P1
            step = 0
            while True:
                if step != 0 and ((idx == 0 and step % 2 == 0) or (idx == 1 and step % 2 == 1) or idx == 2):
                    for action, _ in agent(game.state):
                        if game(*action, player):
                            break
                else:
                    action = random.choice(game.action_space)
                    while not game(*action, player):
                        action = random.choice(game.action_space)

                # Check termination
                if (winner := game.has_winner()) is not None or game.is_full():
                    result[winner] += 1
                    break

                # Proceed to the next player
                player = Symbol.P2 if player == Symbol.P1 else Symbol.P1
                step += 1
        print(result, end="\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_path", type=str)
    parser.add_argument("-s", "--size", type=int, default=3)
    parser.add_argument("-e", "--epochs", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    test(args.pkl_path, size=args.size, epochs=args.epochs, seed=args.seed)
