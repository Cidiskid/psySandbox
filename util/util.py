import os
from random import normalvariate as Norm
from random import uniform, paretovariate
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


def softmax(items):
    if (type(items) is list):
        sume = sum([exp(key) for key in items])
        return [exp(key) / sume for key in items]
    else:
        sume = sum([exp(items[key]) for key in items])
        return {key: exp(items[key]) / sume for key in items}


def softmaxM1(items):
    if (type(items) is list):
        sume = sum([exp(key) - 1 for key in items])
        return [(exp(key) - 1) / sume for key in items]
    else:
        sume = sum([exp(items[key]) - 1 for key in items])
        return {key: (exp(items[key]) - 1) / sume for key in items}


def random_choice(items):
    if (type(items) == dict):
        sume = sum([items[key] for key in items])
        rand = uniform(0, 1.0) * sume
        for key in items:
            if (rand <= items[key]):
                return key
            rand -= items[key]
        for key in items:
            return key
    elif (type(items) == list):
        assert (len(items) > 0)
        rand = uniform(0, 1.0) * sum(items)
        for i in range(len(items)):
            if (rand <= items[i]):
                return i
            rand -= items[i]
        return len(items) - 1


def max_choice(items):
    assert (len(items) > 0)
    if (type(items) == dict):
        max_key = None
        for key in items:
            if (max_key is None or items[max_key] < items[key]):
                max_key = key
        return max_key
    elif (type(items) == list):
        max_i = 0
        for i in range(len(items)):
            if (items[max_i] <= items[i]):
                max_i = i
        return max_i


def listnorm(items):
    avg = sum(items) / len(items)
    std = (sum([(x - avg) ** 2 for x in items]) / len(items)) ** 0.5
    if (std <= 0): std = 1
    return [(x - avg) / std for x in items]


def norm_softmax(items):
    return softmax(listnorm(items))


def rand_shrink(num, ratio):
    assert num >= -1
    assert num <= 1
    assert ratio < 0.8
    ret = num
    if num > 0:
        ret = num * abs(clip_rsmp(-0.999, 0.999, Norm, mu=ratio, sigma=0.1))

    return ret
