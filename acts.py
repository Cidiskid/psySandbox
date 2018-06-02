# -*- coding:utf-8 -*-
import os
import logging
from random import sample, uniform
from copy import deepcopy
from math import exp
from env import Area, get_area_sample_distr
from util.util import max_choice, norm_softmax


def act_zybkyb(env, agent, T, Tfi):
    try_list = sample(range(env.N), env.arg['ACT']['xdzx']['n_near'])
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
    kT0 = env.arg['ACT']['xdzx']['kT0']
    cd = env.arg['ACT']['xdzx']['cool_down']
    if (dE > 0 or exp(dE / (kT0 * cd ** (T + Tfi))) > uniform(0, 1)):
        #        logging.debug("dE:{}, k:{}, p:{}".format(dE, (kT0 * cd ** (T + Tfi)), exp(dE / (kT0 * cd ** (T + Tfi)))))
        agent.state_now = state_next
    #        agent.RenewRsInfo(agent.state_now,env.getValue(agent.state_now, T),T)
    return agent


def act_tscs(env, agent, T, Tfi):  # 探索尝试
    pass


def act_hqxx(env, agent, T, Tfi):  # 获取信息
    mask_t_id = sample(range(env.N), env.arg["ACT"]["hqxx"]["mask_n"])
    mask_t = [False for i in range(env.N)]
    for i in mask_t_id:
        mask_t[i] = True
    jump_d = env.arg['ACT']['hqxx']['dist']
    state_t = agent.state_now
    for i in range(jump_d):
        state_t = state_t.walk(sample(mask_t_id, 1)[0], sample([-1, 1], 1)[0])
    new_area = Area(state_t, mask_t, env.arg['ACT']['hqxx']['dist'])
    new_area.info = get_area_sample_distr(env=env, area=new_area, T=T, state=agent.state_now,
                                          sample_num=env.arg['ACT']['hqxx']['sample_n'],
                                          dfs_r=env.arg['ACT']['hqxx']['dfs_p'])
    agent.frame_arg['PSM']['m-info'].append(new_area)
    if (not 'max' in agent.inter_area.info or agent.inter_area.info['max'] < new_area.info['max']):
        agent.inter_area = new_area
    return agent


def plan_finish(env, state_from, state_to):
    state_t = state_from
    plan = []
    for i in range(env.N):
        if (state_t[i] == state_to[i]):
            continue
        if (abs(state_to[i] - state_t[i] + env.P) % env.P <= abs(state_t[i] - state_to[i] + env.P) % env.P):
            while (state_t[i] != state_to[i]):
                state_t.walk(i, 1)
                plan.append(state_t)
        else:
            while (state_t[i] != state_to[i]):
                state_t.walk(i, -1)
                plan.append(state_t)
    return plan


def act_jhnd(env, agent, T, Tfi):  # 计划拟定
    sample_states = agent.inter_area.sample_near(state=agent.inter_area.center,
                                                 sample_num=env.arg['ACT']['jhnd']['sample_num'],
                                                 dfs_r=env.arg['ACT']['jhnd']['dfs_r'])
    max_state = None
    max_value = 0
    for state in sample_states:
        t_value = agent.agent_arg['ob'](env.getValue(state, T))
        if (max_state is None or t_value > max_value):
            max_state, max_value = state, t_value
    plan = {"plan": plan_finish(env, agent.state_now, max_state),
            'aim_value': max_value}
    agent.frame_arg['PSM']['m-plan'].append(plan)
    return agent


def act_jhjc(env, agent, T, Tfi):  # 计划决策
    best_plan = -1
    best_value = 0
    for i in range(len(agent.frame_arg['PSM']['m-plan'])):
        concat_plan = plan_finish(env, agent.state_now, agent.frame_arg['PSM']['m-plan']['plan'][0])[:-1]
        agent.frame_arg['PSM']['m-plan']['plan'] = concat_plan + agent.frame_arg['PSM']['m-plan']['plan']
        t_value = env.arg[' ']['jhjc']["plan_eval"](agent.frame_arg['PSM']['m-plan'][i]["aim_value"],
                                                      len(agent.frame_arg['PSM']['m-plan'][i]["aim_plan"]))
        if (best_plan == -1 or t_value > best_value):
            best_plan, best_value = i, t_value
    org_plan_value = -1
    if (not agent.frame_arg['PSM']['a-plan'] is None):
        org_plan_value = env.arg['ACT']['jhjc']["plan_eval"](agent.frame_arg['PSM']['a-plan']["aim_value"],
                                                             len(agent.frame_arg['PSM']['a-plan']["aim_plan"]))
    if (best_value > org_plan_value):
        agent.frame_arg['PSM']['a-plan'] = agent.frame_arg['PSM']['m-plan'][best_plan]
    return agent


def act_jhzx(env, agent, T, Tfi):  # 计划执行
    assert (len(agent.frame_arg['PSM']['a-plan']['plan']) > 0)
    agent.state_now = agent.frame_arg['PSM']['a-plan']['plan'][0]
    #    agent.RenewRsInfo(agent.state_now, env.getValueFromStates(agent.state_now, T), T)
    del agent.frame_arg['PSM']['a-plan']['plan'][0]
    if (len(agent.frame_arg['PSM']['a-plan']['plan']) <= 0):
        agent.frame_arg['PSM']['a-plan'] = None
    return agent
