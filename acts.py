import os
import logging
from random import sample, uniform
from copy import deepcopy
from math import exp
import env
from util.util import max_choice, norm_softmax


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

def act_xdzx(env, agent, T):
    pass

def act_tscs(env, agent, T):
    pass


def act_jhzx(env, agent, T):
    assert (len(agent.frame_arg['SSM']['rs-plan']) > 0)
    agent.state_now = agent.frame_arg['SSM']['rs-plan'][0]
    agent.RenewRsInfo(agent.state_now, env.getValueFromStates(agent.state_now, T), T)
    del agent.frame_arg['SSM']['rs-plan'][0]
    return agent
