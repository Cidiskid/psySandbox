# -*- coding:utf-8 -*-
import os
import logging
from random import sample, uniform
from copy import deepcopy
from math import exp
from env import Area, get_area_sample_distr
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


def act_xdzx(env, agent, T, Tfi):  # 行动执行
    state_next = agent.inter_area.rand_walk(agent.state_now)
    value_now = env.getValue(agent.state_now, T)
    value_next = agent.agent_arg['ob'](env.getValue(state_next, T))
    dE = value_next - value_now
    kT0 = agent.frame_arg['ACT']['xdzx']['kT0']
    cd = agent.frame_arg['ACT']['xdzx']['cool_down']
    if (dE > 0 or exp(dE / (kT0 * cd ** (T + Tfi))) > uniform(0, 1)):
        #        logging.debug("dE:{}, k:{}, p:{}".format(dE, (kT0 * cd ** (T + Tfi)), exp(dE / (kT0 * cd ** (T + Tfi)))))
        agent.state_now = state_next
#        agent.RenewRsInfo(agent.state_now,env.getValue(agent.state_now, T),T)
    return agent


def act_tscs(env, agent, T, Tfi):  # 探索尝试
    pass


def act_hqxx(env, agent, T, Tfi):  # 获取信息
    mask_t_id = sample(range(env.N), agent.frame_arg["ACT"]["hqxx"]["mask_n"])
    mask_t = [False for i in range(env.N)]
    for i in mask_t_id:
        mask_t[i] = True
    jump_d = agent.frame_arg['ACT']['hqxx']['dist']
    state_t = agent.state_now
    for i in range(jump_d):
        state_t = state_t.walk(sample(mask_t_id, 1)[0], sample([-1, 1], 1)[0])
    new_area = Area(state_t, mask_t, agent.frame_arg['ACT']['hqxx']['dist'])
    new_area.info = get_area_sample_distr(env=env, area=new_area, T=T, state=agent.state_now,
                                          sample_num=agent.frame_arg['ACT']['hqxx']['sample_n'],
                                          dfs_r=agent.frame_arg['ACT']['hqxx']['dfs_p'])
    agent.frame_arg['PSM']['m-info'].append(new_area)
    if(not 'max' in agent.inter_area.info or agent.inter_area.info['max'] < new_area.info['max']):
        agent.inter_area = new_area
    return agent

def act_jhnd(env, agent, T, Tfi):  # 计划拟定
    pass


def act_jhjc(env, agent, T, Tfi):  # 计划决策
    pass


def act_jhzx(env, agent, T, Tfi):  # 计划执行
    assert (len(agent.frame_arg['SSM']['rs-plan']) > 0)
    agent.state_now = agent.frame_arg['SSM']['rs-plan'][0]
    agent.RenewRsInfo(agent.state_now, env.getValueFromStates(agent.state_now, T), T)
    del agent.frame_arg['SSM']['rs-plan'][0]
    return agent
