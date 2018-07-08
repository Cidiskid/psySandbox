# -*- coding:utf-8 -*-
from util.config import all_config
from util.util import random_choice
import logging
from agent import Agent, Plan
from env import Env, Area
from group import SoclNet
import acts


def meeting_xtfg(env, agents, member, host, socl_net, record, T, Tfi):  # 协调分工
    assert isinstance(env, Env) and isinstance(socl_net, SoclNet)
    assert isinstance(host, set) and isinstance(member, set)
    assert host.issubset(member)
    cent_weigh = socl_net.get_power_out_close_centrality()
    # cent_weigh = socl_net.get_power_out_degree_centrality()  # 改为od
    plan_pool = []
    # 获取所有host加权后的计划得分的比值
    for x in member:
        agents[x].meeting_now = 'xtfg'  # 添加当前行动记录

    for x in host:
        if not agents[x].a_plan is None:
            plan_v = env.arg['plan']['eval'](agents[x].a_plan.len_to_finish(agents[x].a_plan.area.center),
                                             agents[x].a_plan.goal_value)
            plan_pool.append({"id": x,
                              "plan": agents[x].a_plan,
                              "weight": cent_weigh[x] * plan_v})
    w_sum = sum([pair['weight'] for pair in plan_pool])
    sample_pool = [plan_pool[i]['weight'] / w_sum for i in range(len(plan_pool))]

    for x in member:
        to_commit = plan_pool[random_choice(sample_pool)]['plan']
        socl_net, agents[x] = acts.act_commit(env, socl_net, x, agents[x], record, T, Tfi, to_commit,
                                              member)  # 传入member，以便加入plan.info，后续执行时增强关系
        # socl_net,agents[x] = acts.act_commit(env, socl_net, x, agents[x],record, T, Tfi, to_commit)

    for u in member:
        for v in member:
            if u > v:
                socl_net.relat[u][v]['weight'] = agents[u].agent_arg['re_incr_g'](socl_net.relat[u][v]['weight'])

    return agents, socl_net


def meeting_xxjl(env, agents, member, host, socl_net, record, T, Tfi):  # 信息交流
    assert isinstance(env, Env) and isinstance(socl_net, SoclNet)
    assert isinstance(host, set) and isinstance(member, set)
    assert host.issubset(member)
    ret_info = []
    logging.debug("memeber:%s,host:%s" % (member, host))
    for x in member:
        logging.debug("agent%s meeting_now" % x)
        agents[x].meeting_now = 'xxjl'  # 添加当前行动记录

    for x in member:
        if not x in host:
            socl_net, agents[x] = acts.act_hqxx(env, socl_net, x, agents[x], record, T, Tfi)
    for x in member:
        ret_info += agents[x].get_latest_m_info(env.arg['meeting']['xxjl']['last_p_t'],
                                                env.arg['meeting']['xxjl']['max_num'])
    for x in member:
        agents[x].renew_m_info_list(ret_info, T + Tfi)

    for u in member:
        for v in member:
            if u > v:
                socl_net.relat[u][v]['weight'] = agents[u].agent_arg['re_incr_g'](socl_net.relat[u][v]['weight'])

    return agents, socl_net


def meeting_tljc(env, agents, member, host, socl_net, record, T, Tfi):  # 讨论决策
    assert isinstance(member, set) and isinstance(host, set)
    assert host.issubset(member)
    assert isinstance(agents[0], Agent)
    # host的max_area中，最好的区域用于制定计划
    all_max_area = [agents[x].get_max_area() for x in host]
    max_area = max(all_max_area, key=lambda a: a.info['max'])

    for x in member:
        agents[x].meeting_now = 'tljc'  # 添加当前行动记录

    new_plans = []
    for x in member:
        new_plans.append(acts._act_jhnd_get_plan(env, agents[x], max_area))
        new_plans[-1].info['owner'] = x
        new_plans[-1].info['T_gen'] = T + Tfi
    fin_plan = max(new_plans, key=lambda plan: plan.goal_value)  # 仅根据goal_value排序，不考虑距离

    for x in member:
        socl_net, agents[x] = acts.act_jhjc(env, socl_net, x, agents[x], record, T, Tfi, fin_plan)

    for u in member:
        for v in member:
            if u > v:
                socl_net.relat[u][v]['weight'] = agents[u].agent_arg['re_incr_g'](socl_net.relat[u][v]['weight'])

    return agents, socl_net


meet_map = {
    "xtfg": meeting_xtfg,  # 协调分工
    "xxjl": meeting_xxjl,  # 信息交流
    "tljc": meeting_tljc  # 讨论决策
}
