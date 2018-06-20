# -*- coding:utf-8 -*-
import env
import agent
import arg
import acts
from util import config, moniter, util
from group import SoclNet
from env import Env
from agent import Agent
import logging
from random import uniform
from copy import deepcopy

global_arg = arg.init_global_arg()


def hqxx_zyzx(env, agent, Ti, Tfi, agent_no):
    if (agent.frame_arg['PROC']['action']):
        if (env.getValue(agent.state_now, Ti) >= agent.inter_area.info['max'] * 0.99):
            logging.debug("Agent %d, act_hqxx" % (agent_no))
            agent = acts.act_hqxx(env, agent, Ti, Tfi)
        else:
            logging.debug("Agent %d, act_zyzx" % (agent_no))
            agent = acts.act_zyzx(env, agent, Ti, Tfi)
    return agent


# 仅自由执行，作为baseline
def only_zyzx(env, agent, Ti, Tfi, agent_no):
    if (agent.frame_arg['PROC']['action']):
        logging.debug("Agent %d, act_zyzx" % (agent_no))
        agent = acts.act_zyzx(env, agent, Ti, Tfi)
    return agent


def single_ver1(env, agent, Ti, Tfi, agent_no):
    pass


# 单人版的行动逻辑
def sgl_agent_act(env, soc_net, agent, record, Ti, Tfi, agent_no, meet_req):
    assert isinstance(env, Env) and isinstance(soc_net, SoclNet) and isinstance(agent, Agent)
    # P2-01 添加对arg['PROC']['action']的判断
    if not agent.frame_arg["PROC"]['action']:
        return agent, soc_net, None
    # 各种选项的概率
    # P1-06,07，P2-02 增加新act类型
    dF = agent.get_max_area().info['max'] - env.getValue(agent.state_now)
    prob = {"hqxx_xxjl": agent.frame_arg['ACT']['odds']['hqxx'](dF),
            "jhjc_tljc": agent.frame_arg['ACT']['odds']['jhjc'](dF),
            "xdzx_xtfg": agent.frame_arg['ACT']['odds']['xdzx'](dF)
            }

    use_police = util.random_choice(prob)  # 根据概率参数随机选择一种行动策略
    meet_info = None
    #    logging.debug("meet_req:{}".format(meet_req))
    #    logging.debug("agent_no:{}".format(agent_no))
    if use_police == "hqxx_xxjl":
        last_agent = deepcopy(agent)
        agent.meeting_now = ''  # 不参加会议
        soc_net, agent = acts.act_hqxx(env, soc_net, agent_no, agent, record, Ti, Tfi)
        # 如果获得比之前更好的区域，考虑召集会议一起制定计划，进行讨论决策tljc
        return agent, soc_net, meet_info
    elif use_police == "jhjc_tljc":
        last_agent = deepcopy(agent)
        agent.meeting_now = ''  # 不参加会议
        (soc_net, agent) = acts.act_jhnd(env, soc_net, agent_no, agent, record, Ti, Tfi)
        return agent, soc_net, meet_info
    elif use_police == "xdzx_xtfg":
        last_agent = deepcopy(agent)
        agent.meeting_now = ''  # 不参加会议
        (soc_net, agent) = acts.act_xdzx(env, soc_net, agent_no, agent, record, Ti, Tfi)

        return agent, soc_net, meet_info


# 多人版的行动逻辑
def mul_agent_act(env, soc_net, agent, record, Ti, Tfi, agent_no, meet_req):
    assert isinstance(env, Env) and isinstance(soc_net, SoclNet) and isinstance(agent, Agent)
    # P2-01 添加对arg['PROC']['action']的判断
    if not agent.frame_arg["PROC"]['action']:
        return agent, soc_net, None
    # 各种选项的概率
    # P1-06,07，P2-02 增加新act类型
    dF = agent.get_max_area().info['max'] - env.getValue(agent.state_now)
    prob = {"hqxx_xxjl": agent.frame_arg['ACT']['odds']['hqxx'](dF),
            "jhjc_tljc": agent.frame_arg['ACT']['odds']['jhjc'](dF),
            "xdzx_xtfg": agent.frame_arg['ACT']['odds']['xdzx'](dF),
            "whlj": agent.frame_arg['ACT']['odds']['whlj'](dF),
            "dyjs": agent.frame_arg['ACT']['odds']['dyjs'](dF),
            "tjzt": agent.frame_arg['ACT']['odds']['tjzt'](dF)}

    use_police = util.random_choice(prob)  # 根据概率参数随机选择一种行动策略
    meet_info = None
    #    logging.debug("meet_req:{}".format(meet_req))
    #    logging.debug("agent_no:{}".format(agent_no))

    self_efficacy = soc_net.power[agent_no][agent_no]['weight']
    host_Cod = soc_net.get_power_out_degree_centrality()[agent_no]
    host_Cc = soc_net.get_relat_close_centrality()[agent_no]
    max_relat = {m_name: max([soc_net.relat[x][agent_no]['weight'] for x in meet_req[m_name]])
                 for m_name in meet_req}
    max_power = {m_name: max([soc_net.power[x][agent_no]['weight'] for x in meet_req[m_name]])
                 for m_name in meet_req}

    # 如果选择的策略正好有相应会议，考虑要不要参加
    if use_police == "hqxx_xxjl":
        if 'xxjl' in meet_req:
            # p-cmt：接受概率
            commit_p = agent.frame_arg["ACT"]["p-cmt"]["xxjl"](max_relat['xxjl'], max_power['xxjl'], self_efficacy)
            if (commit_p > uniform(0, 1)):
                return agent, soc_net, {"type": "commit", "name": "xxjl"}
        # 没有会议，直接进行单人的获取信息hqxx行动
        last_agent = deepcopy(agent)
        agent.meeting_now = ''  # 不参加会议
        soc_net, agent = acts.act_hqxx(env, soc_net, agent_no, agent, record, Ti, Tfi)
        # 如果获得比之前更好的区域，考虑召集会议一起制定计划，进行讨论决策tljc
        if (last_agent.get_max_area().info['max'] < agent.get_max_area().info['max']):
            # P1-02 同样用lambda表达式传回
            p_req_tljc = agent.frame_arg["ACT"]["p-req"]["tljc"](self_efficacy, host_Cc, host_Cod)
            if p_req_tljc > uniform(0, 1):
                agent.meeting_now = "tljc_req"  # 发起讨论决策会议
                meet_info = {"type": "req", "name": "tljc"}
        # 如果没有获得更好的区域，考虑召集会议进行信息交流xxjl
        else:
            p_req_xxjl = agent.frame_arg["ACT"]["p-req"]["xxjl"](self_efficacy, host_Cc, host_Cod)
            if p_req_xxjl > uniform(0, 1):
                agent.meeting_now = 'xxjl_req'  # 发起信息交流会议
                meet_info = {"type": "req", "name": "xxjl"}
        return agent, soc_net, meet_info
    elif use_police == "jhjc_tljc":
        if 'tljc' in meet_req:
            p_cmt = agent.frame_arg["ACT"]['p-cmt']['tljc'](max_relat["tljc"], max_power['tljc'], self_efficacy)
            if (p_cmt > uniform(0, 1)):
                return agent, soc_net, {"type": "commit", "name": "tljc"}
        last_agent = deepcopy(agent)
        agent.meeting_now = ''  # 不参加会议
        (soc_net, agent) = acts.act_jhnd(env, soc_net, agent_no, agent, record, Ti, Tfi)
        if (last_agent.a_plan is None or last_agent.a_plan.goal_value < agent.a_plan.goal_value):
            p_req = agent.frame_arg["ACT"]['p-req']['xtfg'](self_efficacy, host_Cc, host_Cod)
            if p_req > uniform(0, 1):
                agent.meeting_now = 'xtfg_req'  # 发起信息交流会议
                meet_info = {"type": "req", "name": "xtfg"}
        return agent, soc_net, meet_info
    elif use_police == "xdzx_xtfg":
        if 'xtfg' in meet_req:
            p_cmt = agent.frame_arg["ACT"]['p-cmt']['xtfg'](max_relat["xtfg"], max_power['xtfg'], self_efficacy)
            if (p_cmt > uniform(0, 1)):
                return agent, soc_net, {"type": "commit", "name": "xtfg"}
        last_agent = deepcopy(agent)
        agent.meeting_now = ''  # 不参加会议
        (soc_net, agent) = acts.act_xdzx(env, soc_net, agent_no, agent, record, Ti, Tfi)

        # 如果计划执行完了(没计划)，或新的计划比原来好，召集讨论决策？
        # 行动执行完后发起讨论感觉有点奇怪，我先删掉这一段
        # if ((last_agent.a_plan is None) or (
        #        not agent.a_plan is None and last_agent.a_plan.goal_value < agent.a_plan.goal_value)):
        #    if soc_net.power[agent_no][agent_no]['weight'] > uniform(0, 1):
        #        meet_info = {"type": "req", "name": "tljc"}

        return agent, soc_net, meet_info
    elif use_police == "whlj":
        agent.meeting_now = ''  # 不参加会议
        soc_net, agent = acts.act_whlj(env, soc_net, agent_no, agent, record, Ti, Tfi)
        return agent, soc_net, meet_info

    elif use_police == "dyjs":
        agent.meeting_now = ''  # 不参加会议
        soc_net, agent = acts.act_dyjs(env, soc_net, agent_no, agent, record, Ti, Tfi)
        return agent, soc_net, meet_info

    elif use_police == "tjzt":
        #  P2-02
        agent.meeting_now = ''  # 不参加会议
        soc_net, agent = acts.act_tjzt(env, soc_net, agent_no, agent, record, Ti, Tfi)
        return agent, soc_net, meet_info


def mul_agent_meet():
    pass


using_brain = [hqxx_zyzx, hqxx_zyzx, hqxx_zyzx, hqxx_zyzx, hqxx_zyzx]

act_brain = [mul_agent_act] * global_arg['Nagent']
meet_brain = [mul_agent_meet] * global_arg['Nagent']
