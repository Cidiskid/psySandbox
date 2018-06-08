# -*- coding:utf-8 -*-
from util.config import all_config
from util.util import random_choice
import logging
from agent import Agent, Plan
from env import Env, Area
from group import SoclNet
import acts


def meeting_xtfg(env, agents, member, host, socl_net, T, Tfi):  # 协调分工
    assert isinstance(env, Env) and isinstance(socl_net, SoclNet)
    assert isinstance(host, set) and isinstance(member, set)
    assert host.issubset(member)
    cent_weigh = socl_net.get_power_close_centrality()
    plan_pool = []
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
        agents[x] = acts.act_commit(env, agents[x], T, Tfi, to_commit)

    return agents, socl_net


def meeting_xxjl(env, agents, member, host, socl_net, T, Tfi):  # 信息交流
    assert isinstance(env, Env) and isinstance(socl_net, SoclNet)
    assert isinstance(host, set) and isinstance(member, set)
    assert host.issubset(member)
    ret_info = []
    for x in member:
        ret_info += agents[x].get_latest_m_info(env.arg['meeting']['xxjl']['last_p_t'],
                                                env.arg['meeting']['xxjl']['max_num'])
    for x in member:
        agents[x].renew_m_info_list(ret_info, T + Tfi)
    return agents, socl_net


def meeting_tljc(env, agents, member, host, socl_net, T, Tfi):  # 讨论决策
    assert isinstance(member, set) and isinstance(host, set)
    assert host.issubset(member)

    all_max_area = [agents[id].get_max_area(x) for x in host]
    max_area = max(all_max_area, key=lambda a: a.info['max'])
    new_plans = [acts._act_jhnd_get_plan(env, agents[x], max_area) for x in host]

    fin_plan = max(new_plans, key=lambda plan: plan.goal_value)

    for x in member:
        agents[x] = acts.act_jhcj(env, agents[x], T, Tfi, fin_plan)
    return agents, socl_net
