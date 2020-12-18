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


import yaml
import world


if __name__ == "__main__":
    with open("conf.yaml", "r") as file:
        conf = yaml.safe_load(file)

    print(
        world.World(
            conf["world"]["size"],
            x_start=conf["world"]["start"]["x"],
            y_start=conf["world"]["start"]["y"],
            x_end=conf["world"]["end"]["y"],
            y_end=conf["world"]["end"]["y"],
        )
    )
