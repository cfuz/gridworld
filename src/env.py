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


import gym
import numpy

from gym import error, spaces, utils
from gym.utils import seeding
from enum import Enum

from agent import Agent
from world import World
from coord import Coord
from cell import Cell
from action import Action


class GridWorld(gym.Env):
    def __init__(self, cfg):
        self.cfg = cfg

        # Action enumeration for this environment
        self.actions = Action
        self.n_actions = len(Action)

        # Generate a new grid at the start of each episode
        self._gen_world()

        self._gen_dist()

        # Step count since episode start
        self.n_steps = 0
        self.step_max = self.cfg["max_steps"]

        # Initialize the RNG
        self.seed(seed=self.cfg["seed"])

    def inject(self, agent: Agent):
        self.world._inject(agent)

    def seed(self, seed=7878):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def render(self, close=False):
        # Render the environment to the screen
        print(self.world, end="")

    def step(
        self, state: int or Coord, action: int or Action
    ) -> (Coord, float, bool, bool):
        if isinstance(action, int) or isinstance(action, numpy.int32):
            action = self.actions.from_idx(action)
        self.n_steps += 1
        next_state, reward, done = self.world.process_action(state, action)
        # if self.n_steps >= self.step_max:
        #     done = True
        return next_state, reward, done, self.is_trap(next_state)

    def is_trap(self, state: int or Coord) -> bool:
        if isinstance(state, int):
            state = Coord.from_state(state)
        return self.world.cell(state) == Cell.Trap

    def gen_obs(self):
        return self.world.agent.pos.to_state()

    def _gen_world(self):
        self.world = World(
            self.cfg["world"]["size"],
            trap_conf=self.cfg["world"]["traps"]
            if "traps" in self.cfg["world"]
            else None,
            x_start=self.cfg["world"]["start"]["x"]
            if "x" in self.cfg["world"]["start"]
            else None,
            y_start=self.cfg["world"]["start"]["y"]
            if "y" in self.cfg["world"]["start"]
            else None,
            x_end=self.cfg["world"]["end"]["y"]
            if "x" in self.cfg["world"]["end"]
            else None,
            y_end=self.cfg["world"]["end"]["y"]
            if "y" in self.cfg["world"]["end"]
            else None,
        )

    def _gen_dist(self):
        def next(state: int, action: Action) -> (int, float, bool):
            if self.world.cell(state) == Cell.Goal:
                return (state, 0.0, True)
            else:
                nstate = (
                    Coord(state % self.world.size, state // self.world.size)
                    + action.value
                )
                if action == Action.North:
                    nstate.y = max(nstate.y, 0)
                elif action == Action.South:
                    nstate.y = min(nstate.y, self.world.size - 1)
                elif action == Action.East:
                    nstate.x = min(nstate.x, self.world.size - 1)
                elif action == Action.West:
                    nstate.x = max(nstate.x, 0)
                nstate = nstate.to_state()
                return (
                    nstate,
                    self.world.flat_reward_map[nstate],
                    self.world.cell(nstate) == Cell.Goal,
                )

        def to_coord(pos: int) -> Coord:
            return Coord(pos % self.world.size, pos // self.world.size)

        def transitions_from_state(
            state: int,
        ) -> (numpy.array, numpy.array, numpy.array):
            # st = {}
            reward_transitions = []
            state_transitions = []
            goal_transitions = []
            for a in self.actions.to_indices():
                nstate, nrew, done = next(state, Action.from_idx(a))
                reward_transitions += [nrew]
                state_transitions += [nstate]
                goal_transitions += [done]
                # st[a] = {'state': nstate, 'reward': nrew, 'goal': ngoal)]
            # return st
            return (
                numpy.array(reward_transitions, dtype=numpy.float32),
                numpy.array(state_transitions, dtype=numpy.int32),
                numpy.array(goal_transitions, dtype=bool),
            )

        self.transitions = {"reward": [], "next": [], "goal": []}

        for state in range(self.world.n_states):
            treward, tnext, tgoal = transitions_from_state(state)
            self.transitions["reward"] += [treward]
            self.transitions["next"] += [tnext]
            self.transitions["goal"] += [tgoal]

        self.transitions["reward"] = numpy.array(
            self.transitions["reward"], dtype=numpy.float32
        )
        self.transitions["next"] = numpy.array(
            self.transitions["next"], dtype=numpy.int32
        )
        self.transitions["goal"] = numpy.array(
            self.transitions["goal"], dtype=numpy.bool
        )
