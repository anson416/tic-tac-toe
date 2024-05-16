# -*- coding: utf-8 -*-
# File: game.py

from collections.abc import Iterator
from enum import Enum
from itertools import product
from typing import Optional


class Symbol(Enum):
    EMPTY = " "
    P1 = "X"
    P2 = "O"


class TicTacToe(object):
    def __init__(self, n: int = 3) -> None:
        self._n = n
        self._board = [[Symbol.EMPTY.value for _ in range(n)] for _ in range(n)]

    def __call__(self, row: int, col: int, symbol: Symbol) -> bool:
        # Handle invalid move
        if self.board[row][col] != Symbol.EMPTY.value or symbol == Symbol.EMPTY:
            return False

        # Update board with valid move
        self._board[row][col] = symbol.value
        return True

    def get_all_states(self) -> Iterator[tuple[tuple[str, ...], ...]]:
        for state in product(Symbol, repeat=self.n**2):
            board = []
            for i in range(self.n):
                row = (s.value for s in state[i * self.n : (i + 1) * self.n])
                board.append(tuple(row))
            yield tuple(board)

    def is_full(self) -> bool:
        for row in self.board:
            for symbol in row:
                if symbol != Symbol.EMPTY.value:
                    return False
        return True

    def has_winner(self) -> Optional[str]:
        # Check rows and columns
        for board in (self.board, self._transpose):
            for i in range(self.n):
                if board[i][0] != Symbol.EMPTY.value and len(set(board[i])) == 1:
                    return board[i][0]

        # Check left-to-right diagonal
        if self.board[0][0] != Symbol.EMPTY.value and len(set(self.board[i][i] for i in range(self.n))) == 1:
            return self.board[0][0]

        # Check right-to-left diagonal
        if (
            self.board[0][self.n - 1] != Symbol.EMPTY.value
            and len(set(self.board[i][self.n - i - 1] for i in range(self.n))) == 1
        ):
            return self.board[0][self.n - 1]

        return False

    @staticmethod
    def print_board(board: list[list[str]]) -> None:
        row = []
        for i in range(len(board)):
            row.append("|".join(board[i]))
            if i != len(board) - 1:
                row.append("-" * (2 * len(board) - 1))
        print(*row, sep="\n")

    @property
    def n(self) -> int:
        return self._n

    @property
    def board(self) -> list[list[str]]:
        return self._board

    @property
    def _transpose(self) -> list[list[str]]:
        return [[self.board[i][j] for i in range(self.n)] for j in range(self.n)]

    @property
    def state(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(row) for row in self.board)
