# -*- coding: utf-8 -*-
# File: game.py

from collections.abc import Iterator
from enum import Enum
from functools import cached_property
from itertools import product
from typing import Optional


class Symbol(Enum):
    EMPTY = " "
    P1 = "X"
    P2 = "O"


class TicTacToe(object):
    def __init__(self, size: int = 3) -> None:
        # Check arguments
        if size < 3:
            raise ValueError(f"`size` is expected to be greater than or equal to 3, got {size}.")

        self._size = size
        self.clear()

    def __call__(self, row: int, col: int, symbol: Symbol) -> bool:
        # Handle invalid move
        if self.board[row][col] != Symbol.EMPTY.value or symbol == Symbol.EMPTY:
            return False

        # Update board with valid move
        self._board[row][col] = symbol.value
        return True

    def clear(self) -> None:
        self._board = [[Symbol.EMPTY.value for _ in range(self.size)] for _ in range(self.size)]

    def is_full(self) -> bool:
        for row in self.board:
            for symbol in row:
                if symbol == Symbol.EMPTY.value:
                    return False
        return True

    def has_winner(self) -> Optional[str]:
        # Check rows and columns
        for board in (self.board, self._transpose):
            for i in range(self.size):
                if board[i][0] != Symbol.EMPTY.value and len(set(board[i])) == 1:
                    return board[i][0]

        # Check left-to-right diagonal
        if self.board[0][0] != Symbol.EMPTY.value and len(set(self.board[i][i] for i in range(self.size))) == 1:
            return self.board[0][0]

        # Check right-to-left diagonal
        if (
            self.board[0][self.size - 1] != Symbol.EMPTY.value
            and len(set(self.board[i][self.size - i - 1] for i in range(self.size))) == 1
        ):
            return self.board[0][self.size - 1]

    @staticmethod
    def print_board(board: list[list[str]]) -> None:
        row = []
        for i in range(len(board)):
            row.append("|".join(board[i]))
            if i != len(board) - 1:
                row.append("-" * (2 * len(board) - 1))
        print(*row, sep="\n")

    @property
    def size(self) -> int:
        return self._size

    @property
    def board(self) -> list[list[str]]:
        return self._board

    @property
    def _transpose(self) -> list[list[str]]:
        return [[self.board[i][j] for i in range(self.size)] for j in range(self.size)]

    @property
    def state(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(row) for row in self.board)

    @property
    def state_space(self) -> Iterator[tuple[tuple[str, ...], ...]]:
        for state in product(Symbol, repeat=self.size**2):
            board = []
            for i in range(self.size):
                row = (s.value for s in state[i * self.size : (i + 1) * self.size])
                board.append(tuple(row))
            yield tuple(board)

    @cached_property
    def action_space(self) -> tuple[tuple[int, int], ...]:
        return tuple(product(range(self.size), repeat=2))
