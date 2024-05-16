# -*- coding: utf-8 -*-
# File: agent.py

import pickle
import random
from collections.abc import Iterator
from typing import Any, Optional


class QAgent(object):
    def __init__(self, seed: Optional[int] = None) -> None:
        random.seed(seed)
        self._table: Optional[dict[Any, dict[Any, float]]] = None

    def __call__(self, state: Any) -> Any:
        if self.table is None:
            raise RuntimeError(f"`{self.__class__.__name__}.table` is not yet initialized.")
        return max(self.table[state].items(), key=lambda x: x[1])[0]

    def init_table(self, state_space: Iterator[Any], action_space: Iterator[Any]) -> None:
        action_space = tuple(action_space)
        self._table = {}
        for state in state_space:
            self._table[state] = {action: random.random() for action in action_space}

    def save(self, pkl_path: str) -> None:
        if self.table is None:
            raise RuntimeError(f"`{self.__class__.__name__}.table` is not yet initialized.")
        with open(pkl_path, "wb") as f:
            pickle.dump(self.table, f)

    def load(self, pkl_path: str) -> None:
        with open(pkl_path, "rb") as f:
            self._table = pickle.load(f)

    def update(
        self,
        state: Any,
        action: Any,
        reward: float,
        future_state: Any,
        learning_rate: float = 0.01,
        discount_factor: float = 0.99,
    ) -> None:
        if self.table is None:
            raise RuntimeError(f"`{self.__class__.__name__}.table` is not yet initialized.")
        current_q = self._table[state][action]
        max_future_q = max(self._table[future_state].values())
        self._table[state][action] += learning_rate * (reward + discount_factor * max_future_q - current_q)

    @property
    def table(self) -> Optional[dict[Any, dict[Any, float]]]:
        return self._table
