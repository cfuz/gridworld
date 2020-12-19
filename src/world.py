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


import numpy
import enum

import agent
from coord import Coord as Vec2


class Cell(enum.Enum):
    Empty = 0
    Start = 1
    Trap = 2
    Goal = 3


class World:
    _CELL_REPR = {
        Cell.Empty: " ",
        Cell.Trap: '🪤"',
        Cell.Start: "🏠",
        Cell.Goal: "🧀",
    }

    _AGT_REPR = "🐭"

    def __init__(
        self,
        size: int,
        trap_conf: dict = None,
        x_start: int = 0,
        y_start: int = 0,
        x_end: int = None,
        y_end: int = None,
    ):

        self.size = size
        self.agent_pos = Vec2(x_start, y_start)

        if x_end is None:
            x_end = size - 1

        if y_end is None:
            y_end = size - 1

        self.reward = {
            Cell.Empty: -1,
            Cell.Trap: -2 * (size - 1),
            Cell.Start: -1,
            Cell.Goal: 2 * (size - 1),
        }

        self.grid = [
            [
                Cell.Goal if col == x_end and line == y_end else Cell.Empty
                for col in range(size)
            ]
            for line in range(size)
        ]

        if trap_conf is not None:
            assert "type" in trap_conf and (
                "fixed" == trap_conf["type"] or "random" == trap_conf["type"]
            ), "Expected a trap configuration type ({fixed|random})"
            assert (
                "dist" in trap_conf
            ), """
            Expected a distribution of type dict (random mode) or a list (fixed
            mode)
            """

            if trap_conf["type"] == "random":
                assert (
                    "empty" in trap_conf["dist"] and "trap" in trap_conf["dist"]
                ), """
                Expected a probability distribution of type: 
                trap_conf.dist = { "empty": float in [0.0, 1.0], "trap": float in 
                [0.0, 1.0] }
                """

                ctypes = [Cell.Empty, Cell.Trap]
                dist = [trap_conf["dist"]["empty"], trap_conf["dist"]["trap"]]
                rng_map = [[numpy.random.choice(ctypes, size, p=dist)] for _ in size]

                for lidx, line in enumerate(rng_map):
                    for cidx, cell in enumerate(line):
                        if (lidx == y_start and cidx == x_start) or (
                            lidx == y_end and cidx == x_end
                        ):
                            continue
                        else:
                            self.grid[lidx][cidx] = cell
            else:
                assert (
                    type(trap_conf["dist"]) == list
                ), """
                Expected a distribution of type: 
                trap_conf.dist = [ { "x": int, "y": int }, .. ]
                """

                for coord in trap_conf["dist"]:
                    assert (
                        "x" in coord and "y" in coord
                    ), """
                    Wrong coordinate format. Expected a coordinate of type: 
                    { "x": int, "y": int }
                    """
                    if (coord["x"] != x_end or coord["y"] != y_end) and (
                        coord["x"] != x_start or coord["y"] != y_start
                    ):
                        self.grid[coord["y"]][coord["x"]] = Cell.Trap

    def process_action(self, action: Vec2):
        if self.agent_pos.x + action.x >= 0 and self.agent_pos.x + action.x < self.size:
            if self.agent_pos.y + action.y >= 0 and self.agent_pos.y + action.y < self.size:
                self.agent_pos.x += action.x
                self.agent_pos.y += action.y
                grid_kind = self.grid[self.agent_pos.y][self.agent_pos.x]
                return self.reward[grid_kind]
            

    def reward_grid(self) -> list:
        return [[self.reward[col] for col in line] for line in self.grid]

    def __repr__(self):
        sep: str = f"\033[1;30m{'':5}+"
        head: str = f"\033[1;33m{'':5}"

        for idx in range(len(self.grid)):
            sep += f"{'':-<6}+"
            head += f" {idx:^6}"

        sep += "\033[0m\n"
        head += "\033[0m\n"

        grid_repr: str = head

        for lidx, line in enumerate(self.grid):
            grid_repr += sep
            grid_repr += f"\033[1;33m{lidx:^5}\033[1;30m|\033[0m"

            for cidx, col in enumerate(line):
                elt = (
                    World._AGT_REPR
                    if self.agent_pos == Vec2(cidx, lidx)
                    else World._CELL_REPR[col]
                )
                grid_repr += f"{elt:^{5 if elt != ' ' else 6}}\033[1;30m|\033[0m"

            grid_repr += "\n"

        return grid_repr + sep
