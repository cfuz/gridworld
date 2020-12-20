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


import gym
import numpy

from gym import error, spaces, utils
from gym.utils import seeding
from enum import Enum

from world import World
from coord import Coord
from cell import Cell
from action import Action


class GridWorld(gym.Env):
    """2D grid world game environment"""

    metadata = {
        "render.modes": ["human"],
    }

    def __init__(self, cfg):
        self.cfg = cfg

        # Instanciating world & setting step number to 0
        self.reset()

        # Action enumeration for this environment
        self.actions = Action
        self.idx_to_action = [
            self.actions.North,
            self.actions.East,
            self.actions.South,
            self.actions.West,
        ]

        self.n_actions = len(self.actions)

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.world.n_cells)

        self.dist = self.gen_dist()

        # Number of cells (width and height) in the agent view
        # self.agent_view_size = 1

        self.step_max = self.cfg["max_steps"]

        # Initialize the RNG
        self.seed(seed=self.cfg["seed"])

    def reset(self):
        # Generate a new grid at the start of each episode
        self._gen_world()

        # Step count since episode start
        self.n_steps = 0

        # Return first observation
        obs = self.gen_obs()

        return obs

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human", close=False):
        # Render the environment to the screen
        print(self.world)

    def step(self, action):
        if type(action) == int:
            action = self.idx_to_action[action]

        self.n_steps += 1

        reward, done = self.world.process_action(action.value)

        if self.n_steps >= self.step_max:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}

    def gen_obs(self):
        return self.world.agent_pos.y * self.world.size + self.world.agent_pos.x

    def _gen_world(self):
        self.world = World(
            self.cfg["world"]["size"],
            trap_conf=self.cfg["world"]["traps"],
            x_start=self.cfg["world"]["start"]["x"],
            y_start=self.cfg["world"]["start"]["y"],
            x_end=self.cfg["world"]["end"]["y"],
            y_end=self.cfg["world"]["end"]["y"],
        )

    def gen_dist(self):
        def to_state(row, col):
            return col * self.world.size + row

        def next_pos(row, col, action):
            if action == self.actions.North:
                col = max(col - 1, 0)
            elif action == self.actions.South:
                col = min(col + 1, self.world.size - 1)
            elif action == self.actions.East:
                row = min(row + 1, self.world.size - 1)
            elif action == self.actions.West:
                row = max(row - 1, 0)

            return (row, col)

        def update_dist_matrix(row, col, action):
            action = self.idx_to_action[action]
            new_row, new_col = next_pos(row, col, action)
            new_state = to_state(new_row, new_col)
            new_cell_kind = self.world.cell_at(Coord(new_row, new_col))
            done = new_cell_kind == Cell.Goal
            reward = self.world.reward[new_cell_kind]
            return new_state, reward, done

        dist = {
            s: {a: [] for a in range(self.n_actions)} for s in range(self.world.n_cells)
        }

        for row in range(self.world.size):
            for col in range(self.world.size):
                s = to_state(row, col)

                for a in range(4):
                    li = dist[s][a]
                    ctype = self.world.cell_at(Coord(row, col))

                    if ctype == Cell.Goal:
                        r = self.world.reward[ctype]
                        li.append((1.0, s, r, True))
                    else:
                        li.append((1.0, *update_dist_matrix(row, col, a)))

        return dist
