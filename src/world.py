#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = ["Jarod Duret", "Jonathan Heno"]
__credits__ = ["Jarod Duret", "Jonathan Heno"]
__version__ = "1.0.0"
__maintainer__ = ["Jarod Duret", "Jonathan Heno"]
__email__ = [
    "jarod.duret@alumni.univ-avignon.fr",
    "jonathan.heno@alumni.univ-avignon.fr",
]


import numpy
import enum

from agent import Agent
from coord import Coord
from cell import Cell
from action import Action


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
        Coord._SIZE = size
        self.n_states = size ** 2

        self.cell_width = 7
        self.gui_size = (self.cell_width + 1) * self.size + 6

        self.state = Coord(x_start, y_start)

        if x_end is None:
            x_end = size - 1
        if y_end is None:
            y_end = size - 1

        self._reward = {
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

        self.flat_reward_map = numpy.array(
            [self._reward[cell] for line in self.grid for cell in line],
            dtype=numpy.float32,
        )

    def _inject(self, agent: Agent):
        self.agent = agent

    def reward(self, elt: Coord or Cell) -> int:
        if isinstance(elt, Cell):
            return self._reward[elt]
        elif isinstance(elt, Coord):
            return self._reward[self.grid[elt.y][elt.x]]
        elif isinstance(elt, int):
            assert elt >= 0 and elt < self.n_states, "Index out of range.."
            state = Coord.from_state(elt)
            return self._reward[self.grid[state.y][state.x]]
        else:
            raise f"Unrecognized type for type {elt}.."

    def cell(self, elt: Coord or int) -> Cell:
        if isinstance(elt, Coord):
            return self.grid[elt.y][elt.x]
        elif isinstance(elt, int):
            assert elt >= 0 and elt < self.n_states, "Index out of range.."
            state = Coord.from_state(elt)
            return self.grid[state.y][state.x]
        else:
            raise f"Unrecognized type for type {elt}.."

    def process_action(
        self, state: int or Coord, action: Action
    ) -> (Coord, float, bool):
        if isinstance(state, int):
            state = Coord.from_state(state)
        next_state = state + action.value
        ctype = self.cell(next_state)
        return next_state, self._reward[ctype], ctype == Cell.Goal

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

    def __repr__(self) -> str:
        cell_width = 7
        sep = f"\033[1;30m{'':5}+"
        head = f"\033[1;33m{'':5}"

        for idx in range(len(self.grid)):
            sep += f"{'':-<{cell_width}}+"
            head += f" {idx:^{cell_width}}"

        sep += "\033[0m\n"
        head += "\033[0m\n"

        grid_repr = head

        for lidx, line in enumerate(self.grid):
            grid_repr += sep

            elt_line = f"\033[1;33m{lidx:^5}\033[1;30m|\033[0m"

            if self.agent.is_logic == True:
                values = self.agent.value()
                min_val, max_val = values.min(), values.max()
                val_range = max_val - min_val
                if min_val == max_val:
                    scales = None
                else:
                    scales = [
                        ([min_val, min_val + (val_range * 1 / 5)], "\033[0;31m"),
                        (
                            [
                                min_val + (val_range * 1 / 5),
                                min_val + (val_range * 2 / 5),
                            ],
                            "\033[0;35m",
                        ),
                        (
                            [
                                min_val + (val_range * 2 / 5),
                                min_val + (val_range * 3 / 5),
                            ],
                            "\033[0;37m",
                        ),
                        (
                            [
                                min_val + (val_range * 3 / 5),
                                min_val + (val_range * 4 / 5),
                            ],
                            "\033[0;34m",
                        ),
                        (
                            [min_val + (val_range * 4 / 5), min_val + val_range],
                            "\033[0;36m",
                        ),
                        ([max_val, max_val + 1.0], "\033[1;36m"),
                    ]
                action_line, value_line = (
                    f"{'':<5}\033[1;30m|\033[0m",
                    f"{'':<5}\033[1;30m|\033[0m",
                )

            for cidx, cell in enumerate(line):
                elt = str(cell)
                width = cell_width if elt == "" else cell_width - 1

                if self.agent.pos == Coord(cidx, lidx):
                    elt += f"{self.agent}"
                    width -= 1

                elt_line += f"{elt:^{width}}\033[1;30m|\033[0m"

                if self.agent.is_logic == True:
                    if cell == Cell.Goal:
                        value_line += f"{'':>{cell_width}}"
                        action_line += f"{'':>{cell_width}}"
                    else:
                        value = self.agent.value(Coord(cidx, lidx))
                        if scales is not None:
                            is_max = False
                            for scale, color in scales:
                                if scale[0] <= value < scale[1]:
                                    value_line += f"{color}"
                                    action_line += f"{color}"
                                    is_max = True
                                    break

                        value_line += f"{value:^{cell_width}.3f}"
                        action_line += f"{Action.from_idx(self.agent.policy(Coord(cidx, lidx))):>{cell_width}}"

                    action_line += f"\033[1;30m|\033[0m"
                    value_line += f"\033[1;30m|\033[0m"

            if self.agent.is_logic == True:
                grid_repr += value_line + "\n"

            grid_repr += elt_line + "\n"

            if self.agent.is_logic == True:
                grid_repr += action_line + "\n"

        ep_line = f"{f'EPISODE: {self.agent.n_episodes:>3d}':>{self.gui_size}}\n"
        return grid_repr + sep + ep_line
