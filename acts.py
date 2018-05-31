import os
import logging
from random import sample, uniform
from copy import deepcopy
from math import exp
import env


def listnorm(items):
    avg = sum(items) / len(items)
    std = (sum([(x - avg) ** 2 for x in items]) / len(items)) ** 0.5
    if (std <= 0): std = 1
    return [(x - avg) / std for x in items]


def softmax(items):
    s = sum([exp(x) for x in items])
    return [exp(x) / s for x in items]


def norm_softmax(items):
    return softmax(listnorm(items))


def random_choice(items):
    assert (len(items) > 0)
    rand = uniform(0, 1.0) * sum(items)
    for i in range(len(items)):
        if (rand <= items[i]):
            return i
        rand -= items[i]
    return len(items)-1

def max_choice(items):
    assert (len(items) > 0)
    max_i = 0
    for i in range(len(items)):
        if (items[max_i] <= items[i]):
            max_i = i
    return  max_i

def act_zybkyb(env, agent, T):
    try_list = sample(range(env.N), agent.frame_arg['ACT']['xdzx']['n_near'])
    states = [agent.state_now]
    for x in try_list:
        states.append(deepcopy(agent.state_now))
        states[-1][x] = 1 - int(states[-1][x])
    values = [agent.agent_arg['ob'](env.getValueFromStates(s, T)) for s in states]
#    choice = random_choice(norm_softmax(values))
    choice = max_choice(norm_softmax(values))
    agent.state_now = states[choice]
    agent.RenewRsInfo(agent.state_now,
                      env.getValueFromStates(agent.state_now, T),
                      T)
    return agent


def act_tscs(env, agent, T):
    pass


def act_jhzx(env, agent, T):
    assert (len(agent.frame_arg['SSM']['rs-plan']) > 0)
    agent.state_now = agent.frame_arg['SSM']['rs-plan'][0]
    agent.RenewRsInfo(agent.state_now, env.getValueFromStates(agent.state_now, T), T)
    del agent.frame_arg['SSM']['rs-plan'][0]
    return agent
