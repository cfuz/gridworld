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

from coord import Coord
from cell import Cell


class World:
    def __init__(
        self,
        size: int,
        trap_conf: dict = None,
        x_start: int = 0,
        y_start: int = 0,
        x_end: int = None,
        y_end: int = None,
    ):
        assert (
            x_start >= 0 and x_start < size and y_start >= 0 and y_start < size
        ), "Starting point out of grid.."
        assert (
            x_end >= 0 and x_end < size and y_start >= 0 and y_start < size
        ), "Ending point out of grid.."

        self.size = size
        self.n_cells = size ** 2
        self.agent_pos = Coord(x_start, y_start)

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
                Cell.Goal
                if col == x_end and line == y_end
                else Cell.Start
                if col == x_start and line == y_start
                else Cell.Empty
                for col in range(size)
            ]
            for line in range(size)
        ]
        self.grid

        if trap_conf is not None:
            self.gen_traps(x_start, y_start, x_end, y_end, trap_conf)

    def process_action(self, action: Coord):
        tmp = self.agent_pos + action

        if tmp.x >= 0 and tmp.x < self.size:
            if tmp.y >= 0 and tmp.y < self.size:
                self.agent_pos = tmp

        ctype = self.grid[self.agent_pos.y][self.agent_pos.x]

        return self.reward[ctype], ctype == Cell.Goal

    def gen_traps(
        self, x_start: int, y_start: int, x_end: int, y_end: int, trap_conf: dict
    ):
        assert "type" in trap_conf and (
            "fixed" == trap_conf["type"] or "random" == trap_conf["type"]
        ), "Expected a trap configuration type ({fixed|random})"
        assert (
            "dist" in trap_conf
        ), """
        Expected a distribution of type dict (random mode) or a list (fixed
        mode)
        """

        start, end = Coord(x_start, y_start), Coord(x_end, y_end)

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
            rng_map = [
                numpy.random.choice(ctypes, self.size, p=dist) for _ in range(self.size)
            ]

            for lidx, line in enumerate(rng_map):
                for cidx, col in enumerate(line):
                    pos = Coord(cidx, lidx)
                    if pos != start and pos != end:
                        self.grid[lidx][cidx] = col
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
                pos = Coord(coord["x"], coord["y"])
                if pos != start and pos != end:
                    self.grid[coord["y"]][coord["x"]] = Cell.Trap

    def cell_at(self, pos: Coord) -> Cell:
        return self.grid[pos.y][pos.x]

    def __repr__(self) -> str:
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
                elt = str(col)
                width = 6 if elt == "" else 5

                if self.agent_pos == Coord(cidx, lidx):
                    if col == Cell.Trap:
                        elt += "💀🩹"
                        width -= 2
                    else:
                        elt += "🪖"
                        width -= 1

                grid_repr += f"{elt:^{width}}\033[1;30m|\033[0m"

            grid_repr += "\n"

        return grid_repr + sep
