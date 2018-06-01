import os
import logging
from random import sample, uniform
from copy import deepcopy
from math import exp
import env
from util.util import max_choice, norm_softmax


def act_zybkyb(env, agent, T, Tfi):
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


def act_xdzx(env, agent, T, Tfi):
    state_next = agent.inter_area.rand_walk(agent.state_now)
    value_now = env.getValue(agent.state_now, T)
    value_next = agent.agent_arg['ob'](env.getValue(state_next, T))
    dE = value_next - value_now
    kT0 = agent.frame_arg['ACT']['xdzx']['kT0']
    cd = agent.frame_arg['ACT']['xdzx']['cool_down']
    if (dE > 0 or exp(dE / (kT0 * cd ** (T + Tfi))) > uniform(0, 1)):
#        logging.debug("dE:{}, k:{}, p:{}".format(dE, (kT0 * cd ** (T + Tfi)), exp(dE / (kT0 * cd ** (T + Tfi)))))
        agent.state_now = state_next
        agent.RenewRsInfo(agent.state_now,
                          env.getValue(agent.state_now, T),
                          T)
    return agent


def act_tscs(env, agent, T, Tfi):
    pass


def act_jhzx(env, agent, T, Tfi):
    assert (len(agent.frame_arg['SSM']['rs-plan']) > 0)
    agent.state_now = agent.frame_arg['SSM']['rs-plan'][0]
    agent.RenewRsInfo(agent.state_now, env.getValueFromStates(agent.state_now, T), T)
    del agent.frame_arg['SSM']['rs-plan'][0]
    return agent
