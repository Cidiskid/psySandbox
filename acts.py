# -*- coding:utf-8 -*-
import os
import logging
from random import sample, uniform
from copy import deepcopy
from math import exp
from env import Area, get_area_sample_distr, Env
from group import SoclNet
from util.util import max_choice, norm_softmax, random_choice
from agent import Agent, Plan
from record import Record
import arg


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
def act_zyzx(env, socl_net, agent_no, agent, record, T, Tfi):
    global_area = Area(agent.state_now, [True] * env.N, env.P // 2 * env.N)  # 自由执行自带从global_area找下一个点
    state_next = global_area.rand_walk(agent.state_now)
    value_now = env.getValue(agent.state_now, T)
    value_next = env.arg['ACT']['zyzx']['ob'](env.getValue(state_next, T))  # 改为所有人是一样的ob影响参数
    # agent.agent_arg['ob'](env.getValue(state_next, T))

    dE = value_next - value_now
    kT0 = env.arg['ACT']['xdzx']['kT0']

    cd = env.arg['ACT']['xdzx']['cool_down']
    # cd = env.arg['ACT']['xdzx']['cool_down'] ** (env.arg['P'] ** (env.arg['N'] - inter_area_mskn))
    # 随着时间推移，容忍度越来越低 按stage来衰减
    cd_T = T
    tol = kT0 * cd ** cd_T  # 容忍度
    # logging.debug("tol:{}".format(tol))

    if (dE >= 0 or (tol >= 1e-10 and exp(dE / tol) > uniform(0, 1))):
        agent.state_now = state_next
        new_area = Area(agent.state_now, [False] * env.N, 0)
        new_area.info = get_area_sample_distr(env=env, area=new_area, state=agent.state_now,
                                              T_stmp=T + Tfi, sample_num=1, dfs_r=1)
        # NOTE cid 删除OB扰动，实际到达的点应该给一个客观值
        # for k in new_area.info:
        #   new_area.info[k] = agent.agent_arg['ob'](new_area.info[k])
        agent.renew_m_info(new_area, T + Tfi)

    agent.policy_now = 'zyzx'  # 添加当前行动记录
    logging.debug(agent.policy_now)
    return socl_net, agent


def act_jhzx(env, socl_net, agent_no, agent, record, T, Tfi):  # 计划执行
    assert isinstance(agent.a_plan, Plan)
    # 已经完成原地不动，在area上，向goal移动，在area外向area中心点移动
    agent.state_now = agent.a_plan.next_step(agent.state_now)

    # 将现有的点作为一个区域保存
    new_area = Area(agent.state_now, [False] * env.N, 0)
    new_area.info = get_area_sample_distr(env=env, area=new_area, state=agent.state_now,
                                          T_stmp=T + Tfi, sample_num=1, dfs_r=1)
    # NOTE cid 删除OB扰动，实际到达的点应该给一个客观值
    # for k in new_area.info:
    #    new_area.info[k] = agent.agent_arg['ob'](new_area.info[k])
    agent.renew_m_info(new_area, T + Tfi)
    # 添加当前行动记录
    agent.policy_now = 'jhzx'
    # 计划执行完毕后，清空计划
    if agent.a_plan.is_arrive(agent.state_now):
        #  NOTE cid 添加了清空计划前对计划的评价
        if not agent.a_plan is None:
            dF = env.getValue(agent.state_now) \
                 - record.get_agent_record(agent_no, agent.a_plan.info['T_acpt'])["value"]
            if "commit" in agent.a_plan.info and agent.a_plan.info['commit']:
                dp_f_a = agent.a_plan.info['owner']
                dP_r = agent.agent_arg['dP_r']['other']
            else:
                dp_f_a = agent_no
                dP_r = agent.agent_arg['dP_r']['self']
            dP = agent.agent_arg["dPower"](dF, dP_r)
            d_pwr_updt_g = agent.agent_arg["d_pwr_updt_g"](socl_net.power[dp_f_a][agent_no]['weight'], dP)
            socl_net.power_delta(dp_f_a, agent_no, d_pwr_updt_g)
        agent.policy_now = 'jhzx_fin'

        agent.a_plan = None

    logging.debug(agent.policy_now)
    return socl_net, agent


# 行动执行，T代表该stage第一帧的时间戳，Tfi表示该帧的相对偏移（本stage的第几帧）
def act_xdzx(env, socl_net, agent_no, agent, record, T, Tfi):
    assert isinstance(env, Env) and isinstance(agent, Agent)
    # 如果没计划就自由执行zyzx
    if agent.a_plan is None:
        return act_zyzx(env, socl_net, agent_no, agent, record, T, Tfi)
    assert isinstance(agent.a_plan, Plan)
    if 'commit' in agent.a_plan.info and agent.a_plan.info['commit']:
        # 如果是xtfg分配来的计划，执行时同时触发关系维护机制
        member = agent.a_plan.info['member']
        for u in member:
            if u != agent_no:
                delta_re = agent.agent_arg['d_re_incr_g'](socl_net.relat[u][agent_no]['weight'])
                socl_net.relat_delta(u, agent_no, delta_re)

        return act_jhzx(env, socl_net, agent_no, agent, record, T, Tfi)
    state_val = env.getValue(agent.state_now)
    plan_dist = agent.a_plan.len_to_finish(agent.state_now)
    plan_goal = agent.a_plan.goal_value
    logging.debug("plan_dist: %s" % plan_dist)
    if env.arg['ACT']['xdzx']['do_plan_p'](state_val, plan_dist, plan_goal) > uniform(0, 1):
        return act_jhzx(env, socl_net, agent_no, agent, record, T, Tfi)
    else:
        return act_zyzx(env, socl_net, agent_no, agent, record, T, Tfi)


def act_hqxx(env, socl_net, agent_no, agent, record, T, Tfi):  # 获取信息
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
    new_area.info = get_area_sample_distr(env=env, area=new_area, T_stmp=T + Tfi, state=agent.state_now,
                                          sample_num=env.arg['ACT']['hqxx']['sample_n'],
                                          dfs_r=env.arg['ACT']['hqxx']['dfs_p'])
    # 加ob扰动
    for k in new_area.info:
        new_area.info[k] = agent.agent_arg['ob'](new_area.info[k])

    # 把信息更新到状态中
    agent.renew_m_info(new_area, T + Tfi)
    agent.policy_now = 'hqxx'  # 添加当前行动记录
    logging.debug(agent.policy_now)
    return socl_net, agent


def act_jhjc(env, socl_net, agent_no, agent, record, T, Tfi, new_plan):
    assert isinstance(env, Env) and isinstance(agent, Agent)
    assert isinstance(record, Record) and isinstance(socl_net, SoclNet)
    new_plan_value = env.arg['ACT']['jhjc']["plan_eval"](new_plan.len_to_finish(agent.state_now),
                                                         new_plan.goal_value)
    org_plan_value = env.getValue(agent.state_now)  # TODO cid 如果没有a_plan，至少以当前位置作为plan value
    if not agent.a_plan is None:
        org_plan_value = env.arg['ACT']['jhjc']["plan_eval"](agent.a_plan.len_to_finish(agent.state_now),
                                                             agent.a_plan.goal_value)

    agent.policy_now = 'jhjc_old'  # 添加当前行动记录，维持老计划
    # logging.debug("new_v: %.5s, org_v:%.5s" % (new_plan_value, org_plan_value))
    if new_plan_value >= org_plan_value:
        # P0-07 在覆盖新计划前对旧计划的执行情况进行判断并据此更新SoclNet.power
        if not agent.a_plan is None:
            dF = env.getValue(agent.state_now) \
                 - record.get_agent_record(agent_no, agent.a_plan.info['T_acpt'])["value"]
            if "commit" in agent.a_plan.info and agent.a_plan.info['commit']:
                dp_f_a = new_plan.info['owner']
                dP_r = agent.agent_arg['dP_r']['other']
            else:
                dp_f_a = agent_no
                dP_r = agent.agent_arg['dP_r']['self']
            dP = agent.agent_arg["dPower"](dF, dP_r)
            d_pwr_updt_g = agent.agent_arg["d_pwr_updt_g"](socl_net.power[dp_f_a][agent_no]['weight'], dP)
            socl_net.power_delta(dp_f_a, agent_no, d_pwr_updt_g)
        new_plan.info['T_acpt'] = T + Tfi
        agent.a_plan = new_plan
        logging.debug("plan_dist: %s" % agent.a_plan.len_to_finish(agent.state_now))
        agent.policy_now = 'jhjc_new'  # 添加当前行动记录,选择新计划

    logging.debug(agent.policy_now)
    return socl_net, agent


def act_commit(env, socl_net, agent_no, agent, record, T, Tfi, new_plan, member):
    assert isinstance(env, Env) and isinstance(agent, Agent)
    assert isinstance(socl_net, SoclNet)
    assert isinstance(new_plan, Plan) and isinstance(record, Record)
    # 以owener的power为概率接受一个plan，并且commit
    agent.policy_now = 'commit_f'  # 添加当前行动记录

    if (uniform(0, 1) > socl_net.power[new_plan.info['owner']][agent_no]['weight']):
        # P0-07 同上，覆盖计划前对原计划执行情况进行比较，并更新power
        if not agent.a_plan is None:
            dF = env.getValue(agent.state_now) \
                 - record.get_agent_record(agent_no, agent.a_plan.info['T_acpt'])["value"]
            if "commit" in agent.a_plan.info and agent.a_plan.info['commit']:
                dp_f_a = new_plan.info['owner']
                dP_r = agent.agent_arg['dP_r']['other']
            else:
                dp_f_a = agent_no
                dP_r = agent.agent_arg['dP_r']['self']
            dP = agent.agent_arg["dPower"](dF, dP_r)
            d_pwr_updt_g = agent.agent_arg["d_pwr_updt_g"](socl_net.power[dp_f_a][agent_no]['weight'], dP)
            socl_net.power_delta(dp_f_a, agent_no, d_pwr_updt_g)
        agent.a_plan = deepcopy(new_plan)
        agent.a_plan.info['T_acpt'] = T + Tfi
        agent.a_plan.info['commit'] = True
        agent.a_plan.info['member'] = member
        agent.policy_now = 'commit_t'  # 添加当前行动记录

    logging.debug(agent.policy_now)
    return socl_net, agent


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


def act_jhnd(env, socl_net, agent_no, agent, record, T, Tfi):  # 计划拟定
    assert isinstance(env, Env) and isinstance(agent, Agent)
    max_area = agent.get_max_area()  # TODO WARNING 有可能get一个到达过的局部最优点，不停生成回到这个点的计划
    new_plan = _act_jhnd_get_plan(env, agent, max_area)
    new_plan.info['owner'] = agent_no
    new_plan.info['T_gen'] = T + Tfi
    agent.frame_arg['PSM']['m-plan'].append(new_plan)
    socl_net, agent = act_jhjc(env, socl_net, agent_no, agent, record, T, Tfi, new_plan)
    return socl_net, agent


def act_whlj(env, socl_net, agent_no, agent, record, T, Tfi):
    assert isinstance(socl_net, SoclNet)
    global_arg = arg.init_global_arg()
    to_select = [x for x in range(global_arg['Nagent'])]
    del to_select[agent_no]
    to_whlj = sample(to_select, env.arg['ACT']['whlj']['k'])
    for aim in to_whlj:
        delta = env.arg['ACT']['whlj']['delta_relate'](socl_net.relat[aim][agent_no]['weight'])
        socl_net.relat_delta(aim, agent_no, delta)

    agent.policy_now = 'whlj'  # 添加当前行动记录

    logging.debug(agent.policy_now)
    return socl_net, agent


def act_dyjs(env, socl_net, agent_no, agent, record, T, Tfi):
    assert isinstance(socl_net, SoclNet)
    global_arg = arg.init_global_arg()
    # 根据对自己影响的大小选择强化对象
    out_power = [socl_net.power[x][agent_no]['weight'] for x in range(global_arg["Nagent"])]
    to_power = random_choice(norm_softmax(out_power))
    for aim in range(global_arg['Nagent']):
        delta = env.arg['ACT']['dyjs']['delta_power'](socl_net.power[to_power][aim]['weight'])
        socl_net.power_delta(to_power, aim, delta)

    agent.policy_now = 'dyjs'  # 添加当前行动记录

    logging.debug(agent.policy_now)
    return socl_net, agent


def act_tjzt(env, socl_net, agent_no, agent, record, T, Tfi):
    agent.policy_now = 'tjzt'  # 添加当前行动记录

    logging.debug(agent.policy_now)
    return socl_net, agent
