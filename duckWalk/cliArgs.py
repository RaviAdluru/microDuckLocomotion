"""Turns the TrainConfig dataclass into argparse flags so we never have to keep the two in sync.
"""

import argparse
from dataclasses import fields

from duckWalk.config import TrainConfig


def strToBool(text):
    if text.lower() in ("1", "true", "yes", "y"):
        return True
    if text.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got '{text}'")


def addConfigArgs(parser):
    defaults = TrainConfig()
    for f in fields(TrainConfig):
        value = getattr(defaults, f.name)
        if isinstance(value, bool):
            parser.add_argument(f"--{f.name}", type=strToBool, default=value)
        elif isinstance(value, tuple):
            parser.add_argument(f"--{f.name}", type=int, nargs="+", default=list(value))
        else:
            parser.add_argument(f"--{f.name}", type=type(value), default=value)
    return parser


def configFromArgs(args):
    values = {f.name: getattr(args, f.name) for f in fields(TrainConfig)}
    values["hiddenDims"] = tuple(values["hiddenDims"])
    return TrainConfig(**values)
