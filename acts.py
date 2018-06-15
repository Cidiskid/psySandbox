# -*- coding:utf-8 -*-
import os
import logging
from random import sample, uniform
from copy import deepcopy
from math import exp
from env import Area, get_area_sample_distr, Env
from util.util import max_choice, norm_softmax
from agent import Agent, Plan

# 已经无效
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


# 自由执行，T代表该stage第一帧的时间戳，Tfi表示该帧的相对偏移（本stage的第几帧）
def act_zyzx(env, agent, T, Tfi):
    global_area = Area(agent.state_now, [True] * env.N, env.P // 2 * env.N)
    state_next = global_area.rand_walk(agent.state_now)
    value_now = env.getValue(agent.state_now, T)
    value_next = agent.agent_arg['ob'](env.getValue(state_next, T))

    dE = value_next - value_now
    kT0 = env.arg['ACT']['xdzx']['kT0']

    cd = env.arg['ACT']['xdzx']['cool_down']
    # cd = env.arg['ACT']['xdzx']['cool_down'] ** (env.arg['P'] ** (env.arg['N'] - inter_area_mskn))
    # 随着时间推移，容忍度越来越低
    fake_stage = 16  # 静态测试时，fake一个stage的分隔
    cd_T = Tfi
    tol = kT0 * cd ** cd_T  # 容忍度
    # logging.debug("cd:{}".format(cd))
    # logging.debug("tol:{}".format(tol))

    if (dE >= 0 or (tol >= 1e-10 and exp(dE / tol) > uniform(0, 1))):
        agent.state_now = state_next
        new_area = Area(agent.state_now, [False] * env.N, 0)
        new_area.info = get_area_sample_distr(env=env, area=new_area, state=agent.state_now,
                                              T_stmp=T + Tfi, sample_num=1, dfs_r=1)
        agent.renew_m_info(new_area, T + Tfi)

    return agent


def act_jhzx(env, agent, T, Tfi):  # 计划执行
    assert isinstance(agent.a_plan, Plan)
    # 已经完成原地不动，在计划路径上移动一步，在计划路径外，向中心点移动
    agent.state_now = agent.a_plan.next_step(agent.state_now)

    # 将现有的点作为一个区域保存
    # TODO P1-00和获取信息hqxx的区别：sample的结果是客观结果（能否统一？）
    new_area = Area(agent.state_now, [False] * env.N, 0)
    new_area.info = get_area_sample_distr(env=env, area=new_area, state=agent.state_now,
                                          T_stmp=T + Tfi, sample_num=1, dfs_r=1)
    agent.renew_m_info(new_area, T + Tfi)

    # 计划执行完毕后，清空计划
    if agent.a_plan.is_arrive(agent.state_now):
        agent.a_plan = None
    return agent


# 行动执行，T代表该stage第一帧的时间戳，Tfi表示该帧的相对偏移（本stage的第几帧）
def act_xdzx(env, agent, T, Tfi):
    assert isinstance(env, Env) and isinstance(agent, Agent)
    # 如果没计划就自由执行zyzx
    if agent.a_plan is None:
        return act_zyzx(env, agent, T, Tfi)
    assert isinstance(agent.a_plan, Plan)
    if 'commit' in agent.a_plan.info and agent.a_plan.info['commit']:
        return act_jhzx(env, agent, T, Tfi)
    state_val = env.getValue(agent.state_now)
    plan_dist = agent.a_plan.len_to_finish(agent.state_now)
    plan_goal = agent.a_plan.goal_value
    if env.arg['ACT']['xdzx']['do_plan_p'](state_val, plan_dist, plan_goal) > uniform(0, 1):
        return act_jhzx(env, agent, T, Tfi)
    else:
        return act_zyzx(env, agent, T, Tfi)


def act_hqxx(env, agent, T, Tfi):  # 获取信息
    assert isinstance(env, Env) and isinstance(agent, Agent)
    mask_t_id = sample(range(env.N), env.arg["ACT"]["hqxx"]["mask_n"])  # 随机从N个位点中选择mask_n的位允许修改
    mask_t = [False for i in range(env.N)]
    for i in mask_t_id:
        mask_t[i] = True

    jump_d = env.arg['ACT']['hqxx']['dist']
    state_t = agent.state_now

    # 寻找一个包含当前节点的区域的中心，记为state_t
    for i in range(jump_d):
        state_t = state_t.walk(sample(mask_t_id, 1)[0], sample([-1, 1], 1)[0])

    # 生成一个区域
    new_area = Area(state_t, mask_t, env.arg['ACT']['hqxx']['dist'])

    # 从区域中取样，获取信息，目前支持Max，Min,Avg,Mid,p0.15,p0.85
    # TODO P1-03 考虑改为OB值而不是客观值？
    new_area.info = get_area_sample_distr(env=env, area=new_area, T_stmp=T + Tfi, state=agent.state_now,
                                          sample_num=env.arg['ACT']['hqxx']['sample_n'],
                                          dfs_r=env.arg['ACT']['hqxx']['dfs_p'])
    # 把信息更新到状态中
    agent.renew_m_info(new_area, T + Tfi)
    return agent


def act_jhjc(env, agent, T, Tfi, new_plan):
    assert isinstance(env, Env) and isinstance(agent, Agent)
    new_plan_value = env.arg['ACT']['jhjc']["plan_eval"](new_plan.goal_value,
                                                         new_plan.len_to_finish(agent.state_now))
    org_plan_value = -1
    if not agent.a_plan is None:
        org_plan_value = env.arg['ACT']['jhjc']["plan_eval"](agent.a_plan.goal_value,
                                                             agent.a_plan.len_to_finish(agent.state_now))
    if new_plan_value >= org_plan_value:
        # TODO P0-07 在覆盖新计划前对旧计划的执行情况进行判断并据此更新SoclNet.power
        # not a_plan is None，不为空
        # 提取plan.info中的owner,commit和时间,见P0-08，提取当前时间和计划采纳时间时的适应分数
        # 若commit=ture，对owner的power按权重更新，更新数值与适应分数差有关，具体见文档
        # 若commit=false，更新"自信度"power[i][i]
        # TODO notes: 请refine, 似乎需要传一个socl_net进来,才能update plan owner和agent的关系
        # dF = env.getValue(agent.state_now, T) - FS[agent.a_plan.info['acpt_time']][value]


        # TODO P0-07 为new_plan添加采纳时间戳信息
        agent.a_plan = new_plan

    return agent

# TODO P0-03 修改commit的原理，参照文档
def act_commit(env, agent, T, Tfi, new_plan):
    assert isinstance(env, Env) and isinstance(agent, Agent)
    assert isinstance(new_plan, Plan)
    # TODO notes: 传入一个commit概率，如果commit，将owner信息添加到new_plan中

    # 50% 的概率接受一个plan，并且commit
    if (uniform(0, 1) > 0.5):
        # TODO P0-07 同上，覆盖计划前对原计划执行情况进行比较，并更新power
        agent.a_plan = deepcopy(new_plan)

        # TODO P0-07 为new_plan添加采纳时间戳信息
        agent.a_plan.info['commit'] = True
    return agent


def _act_jhnd_get_plan(env, agent, aim_area):
    sample_states = aim_area.sample_near(state=aim_area.center,
                                         sample_num=env.arg['ACT']['jhnd']['sample_num'],
                                         dfs_r=env.arg['ACT']['jhnd']['dfs_r'])
    max_state = None
    max_value = 0
    for state in sample_states:
        t_value = agent.agent_arg['ob'](env.getValue(state))
        if max_state is None or t_value > max_value:
            max_state, max_value = state, t_value
    new_plan = Plan(goal=max_state, goal_value=max_value, area=aim_area)
    return new_plan


def act_jhnd(env, agent, T, Tfi):  # 计划拟定
    assert isinstance(env, Env) and isinstance(agent, Agent)
    max_area = agent.get_max_area() # TODO notes: 会不会get到一个过去的点？到达过局部最优后，不停生成回到这个点的计划
    new_plan = _act_jhnd_get_plan(env, agent, max_area)
    agent.frame_arg['PSM']['m-plan'].append(new_plan)
    agent = act_jhjc(env, agent, T, Tfi, new_plan)
    return agent
