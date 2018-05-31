import os
from random import normalvariate as Norm
from random import uniform
from math import exp, pow, tanh, cos, pi
from copy import deepcopy


def clip(low, up, fn, *args, **kwargs):
    x = fn(*args, **kwargs)
    if (x > up):
        return up
    if (x < low):
        return low
    return x


def clip_rsmp(low, up, fn, *args, **kwargs):
    while (True):
        x = fn(*args, **kwargs)
        if low <= x <= up:
            return x


def clip_tanh(low, up, c, fn, *args, **kwargs):
    return (tanh(2 * (fn(*args, **kwargs) - c) / (up - low)) + 1) * (up - low) / 2 + low


def softmaxM1(items):
    if (type(items) is list):
        sume = sum([exp(key) - 1 for key in items])
        return [(exp(key) - 1) / sume for key in items]
    else:
        sume = sum([exp(items[key]) - 1 for key in items])
        return {key: (exp(items[key]) - 1) / sume for key in items}


def random_choice(items):
    sume = sum([items[key] for key in items])
    rand = uniform(0, 1.0) * sume
    for key in items:
        if (rand <= items[key]):
            return key
        rand -= items[key]
    for key in items:
        return key


def max_choice(items):
    max_key = None
    for key in items:
        if (max_key is None or items[max_key] < items[key]):
            max_key = key
    return max_key
