#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = ["Jarod Duret", "Jonathan Heno"]
__credits__ = ["Jarod Duret", "Jonathan Heno"]
__version__ = "0.1.0"
__maintainer__ = ["Jarod Duret", "Jonathan Heno"]
__email__ = [
    "jarod.duret@alumni.univ-avignon.fr",
    "jonathan.heno@alumni.univ-avignon.fr",
]


import enum
import agent


class Cell(enum.Enum):
    Empty = 0
    Start = 1
    Trap = 2
    Goal = 3
    Bot = 4


class Action(enum.Enum):
    North = 0
    East = 1
    South = 2
    West = 3


class World:
    _CELL_REPR = {
        Cell.Empty: " ",
        Cell.Bot: "🐭",
        Cell.Trap: "🪤",
        Cell.Start: "🏠",
        Cell.Goal: "🧀",
    }

    def __init__(
        self,
        size: int,
        traps: bool = False,
        x_start: int = 0,
        y_start: int = 0,
        x_end: int = None,
        y_end: int = None,
    ):
        if x_end is None:
            x_end = size - 1

        if y_end is None:
            y_end = size - 1

        if not traps:
            self.grid = [
                [
                    Cell.Goal
                    if col == x_end and line == y_end
                    else Cell.Bot
                    if (line == x_start and col == y_start)
                    else Cell.Empty
                    for col in range(size)
                ]
                for line in range(size)
            ]

        self.agent = agent.RandomAgent(x_start, y_start)
        self.agent.scan(self.grid)

    def reward(self, x: int, y: int) -> int:
        cell = self.grid[x * self.size + y]
        if cell == Cell.Empty:
            return -1
        elif cell == Cell.Trap:
            return -2 * (self.size - 1)
        elif cell == Cell.Goal:
            return 2 * (self.size - 1)
        else:
            return 0

    def __repr__(self):
        sep: str = f"{'':6}+"
        head: str = f"{'':6}"

        for idx in range(len(self.grid)):
            sep += f"{'':-<6}+"
            head += f" {idx:^6}"

        sep += "\n"
        head += "\n"

        grid_repr: str = head

        for idx, line in enumerate(self.grid):
            grid_repr += sep
            grid_repr += f"{idx:^6}|"

            for col in line:
                grid_repr += (
                    f"{World._CELL_REPR[col]:^{5 if col != Cell.Empty else 6}}|"
                )

            grid_repr += "\n"

        return grid_repr + sep
