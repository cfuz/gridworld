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
        grid_size=6,
        max_steps=100,
        seed=1337,
    ):

        # Action enumeration for this environment
        self.actions = GridWorld.Actions
        self.actions_idx = [self.actions.North, self.actions.East, self.actions.South, self.actions.West]

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        self.size = grid_size
        self.grid = World(grid_size)

        # Number of cells (width and height) in the agent view
        #self.agent_view_size = 1

        # Range of possible rewards
        #self.reward_range = (0, 1)

        self.max_steps = max_steps

        # Initialize the RNG
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

    def reset(self):

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.size)

        # These fields should be defined by _gen_grid
        assert self.world.agent_pos is not None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = None # self.gen_obs()
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

        print(action)
        self.step_count += 1

        reward = 0
        done = False

        # Rotate left
        reward = self.world.process_action(action.value)

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}
    
    def gen_obs(self):
        return 1

    def _gen_grid(self, size):
        self.world = World(size)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.randint(xLow, xHigh),
            self.np_random.randint(yLow, yHigh)
        )