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
import coord
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
        if x_end is None:
            x_end = size - 1

        if y_end is None:
            y_end = size - 1

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
                rng_map = [
                    numpy.random.choice(ctypes, size, p=dist) for _ in range(size)
                ]

                for lidx, line in enumerate(rng_map):
                    for cidx, col in enumerate(line):
                        if (lidx == y_start and cidx == x_start) or (
                            lidx == y_end and cidx == x_end
                        ):
                            continue
                        else:
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
                    if (coord["x"] != x_end or coord["y"] != y_end) and (
                        coord["x"] != x_start or coord["y"] != y_start
                    ):
                        self.grid[coord["y"]][coord["x"]] = Cell.Trap

        self.agent = agent.RandomAgent(self.grid, x_start, y_start)

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
                elt = str(col)
                width = 6 if elt == "" else 5
                if self.agent.pos == coord.Coord(cidx, lidx):
                    if col == Cell.Trap:
                        elt += "💀🩹"
                        width -= 2
                    else:
                        elt += str(self.agent)
                        width -= 1
                grid_repr += f"{elt:^{width}}\033[1;30m|\033[0m"

            grid_repr += "\n"

        return grid_repr + sep
