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
from gym import error, spaces, utils
from gym.utils import seeding
from enum import Enum
import numpy as np
from world import World
from coord import Coord as Vec2
from cell import Cell

class GridWorld(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human'],
    }

    # Enumeration of possible actions
    class Actions(Enum):
        North = Vec2(0, -1)
        East = Vec2(1, 0)
        South = Vec2(0, 1)
        West = Vec2(-1, 0)

    def __init__(
        self,
        cfg,
    ):
        self.cfg = cfg
        self.size = self.cfg["world"]["size"]

        # Action enumeration for this environment
        self.actions = GridWorld.Actions
        self.actions_idx = [self.actions.North, self.actions.East, self.actions.South, self.actions.West]

        self.nA = len(self.actions)
        self.nS = self.size * self.size

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.world = World(
            self.cfg["world"]["size"],
            trap_conf=self.cfg["traps"],
            x_start=self.cfg["world"]["start"]["x"],
            y_start=self.cfg["world"]["start"]["y"],
            x_end=self.cfg["world"]["end"]["y"],
            y_end=self.cfg["world"]["end"]["y"]
        )

        self.P = self.gen_P()

        # Number of cells (width and height) in the agent view
        #self.agent_view_size = 1

        self.max_steps = self.cfg["world"]["max_steps"]

        # Initialize the RNG
        self.seed(seed=self.cfg["world"]["seed"])

        # Initialize the state
        self.reset()

    def reset(self):

        # Generate a new grid at the start of each episode
        self._gen_grid()

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        return obs

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(self.world)
        pass

    def step(self, action):
        if type(action) == int:
            action = self.actions_idx[action]

        self.step_count += 1

        reward, done = self.world.process_action(action.value)

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}
    
    def gen_obs(self):
        return self.world.agent_pos.y * self.size + self.world.agent_pos.x

    def _gen_grid(self):
        self.world = World(
            self.cfg["world"]["size"],
            trap_conf=self.cfg["traps"],
            x_start=self.cfg["world"]["start"]["x"],
            y_start=self.cfg["world"]["start"]["y"],
            x_end=self.cfg["world"]["end"]["y"],
            y_end=self.cfg["world"]["end"]["y"]
        )

    def gen_P(self):

        def to_s(row, col):
            return col*self.size + row

        def next_pos(row, col, action):
            if action == self.actions.North:
                col = max(col - 1, 0)
            elif action == self.actions.South:
                col = min(col + 1, self.size - 1)
            elif action == self.actions.East:
                row = min(row + 1, self.size - 1)
            elif action == self.actions.West:
                row = max(row - 1, 0)
            return (row, col)

        def update_p_matrix(row, col, action):
            action = self.actions_idx[action]
            new_row, new_col = next_pos(row, col, action)
            new_state = to_s(new_row, new_col)
            new_cell_kind = self.world.cell_at(Vec2(new_row, new_col))
            done = new_cell_kind == Cell.Goal
            reward = self.world.reward[new_cell_kind]
            return new_state, reward, done

        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        for row in range(self.size):
            for col in range(self.size):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    cell_kind = self.world.cell_at(Vec2(row, col))
                    if cell_kind == Cell.Goal:
                        r = self.world.reward[cell_kind]
                        li.append((1., s, r, True))
                    else:
                        li.append((1., *update_p_matrix(row, col, a)))

        return P