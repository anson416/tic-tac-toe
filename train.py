# -*- coding: utf-8 -*-
# File: train.py

import argparse
import random
from datetime import datetime
from typing import Optional

from tqdm import tqdm

from agent import Agent
from game import Symbol, TicTacToe


def train(
    size: int = 3,
    epochs: int = 1000000,
    learning_rate: float = 0.01,
    discount_factor: float = 0.99,
    random_search: int = 0,
    min_epsilon: float = 0.001,
    epsilon_decay: float = 0.999975,
    seed: Optional[int] = None,
) -> None:
    # Check arguments
    if not (0 <= learning_rate <= 1):
        raise ValueError(f"`learning_rate` is expected to be in range [0, 1], got {learning_rate}.")
    if not (0 <= discount_factor <= 1):
        raise ValueError(f"`discount_factor` is expected to be in range [0, 1], got {discount_factor}.")
    if not (0 <= min_epsilon <= 1):
        raise ValueError(f"`min_epsilon` is expected to be in range [0, 1], got {min_epsilon}.")
    if not (0 <= epsilon_decay <= 1):
        raise ValueError(f"`epsilon_decay` is expected to be in range [0, 1], got {epsilon_decay}.")

    # Seed random number generator
    random.seed(seed)

    # Initialize
    game = TicTacToe(size=size)
    agent = Agent()
    agent.init_table(game.state_space, game.action_space)
    epsilon = 1

    for epoch in tqdm(range(epochs), desc="Training"):
        game.clear()
        updates = []
        player = Symbol.P1
        step = 0
        while True:
            state = game.state
            if step == 0 or epoch < random_search or random.random() < epsilon:
                action = random.choice(game.action_space)
                while not game(*action, player):
                    action = random.choice(game.action_space)
            else:
                actions = agent(state)
                for action, _ in actions:
                    if game(*action, player):
                        break
            updates.append((player.value, state, action, game.state))

            # Check termination
            if (winner := game.has_winner()) is not None or game.is_full():
                break

            # Proceed to the next player
            player = Symbol.P2 if player == Symbol.P1 else Symbol.P1
            step += 1

        # Update Q-table
        for i, update in enumerate(updates, start=1):
            reward = 1 if winner is None else 10 if update[0] == winner else -10
            agent.update(
                *update[1:3],
                reward * i,
                update[-1],
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )

        # Update epsilon
        if epoch >= random_search:
            epsilon = max(epsilon * epsilon_decay, min_epsilon)

    now = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    agent.save(f"qtable_{now}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", type=int, default=3)
    parser.add_argument("-e", "--epochs", type=int, default=1000000)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-df", "--discount_factor", type=float, default=0.99)
    parser.add_argument("-rs", "--random_search", type=int, default=0)
    parser.add_argument("-me", "--min_epsilon", type=float, default=0.001)
    parser.add_argument("-ed", "--epsilon_decay", type=float, default=0.999975)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    train(
        size=args.size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        random_search=args.random_search,
        min_epsilon=args.min_epsilon,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )
