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

# 多人版的行动逻辑
def mul_agent_act(env, soc_net, agent, Ti, Tfi, agent_no, meet_req):
    assert isinstance(env, Env) and isinstance(soc_net, SoclNet) and isinstance(agent, Agent)
    # 各种选项的概率
    prob = {"hqxx_xxjl": 1, "jhjc_tljc": 1, "xdzx_xtfg": 1}
    use_police = util.random_choice(prob)   # 根据概率参数随机选择一种行动策略
    meet_info = None
    #    logging.debug("meet_req:{}".format(meet_req))
    #    logging.debug("agent_no:{}".format(agent_no))

    # 如果选择的策略正好有相应会议，考虑要不要参加
    if use_police == "hqxx_xxjl":
        if 'xxjl' in meet_req:
            # 参与行动的概率为关系最好的召集人的relat值
            max_relat = max([soc_net.relat[x][agent_no]['weight'] for x in meet_req['xxjl']])
            if (max_relat > uniform(0, 1)):
                return agent, {"type": "commit", "name": "xxjl"}
        # 没有会议，直接进行单人的获取信息hqxx行动
        last_agent = deepcopy(agent)
        agent = acts.act_hqxx(env, agent, Ti, Tfi)
        # 如果获得比之前更好的区域，考虑召集会议一起制定计划，进行讨论决策tljc
        if (last_agent.get_max_area().info['max'] < agent.get_max_area().info['max']):
            if soc_net.power[agent_no][agent_no]['weight'] > uniform(0, 1):  # 根据"自信"程度随机选择是否召集会议？
                meet_info = {"type": "req", "name": "tljc"}
        # 如果没有获得更好的区域，考虑召集会议进行信息交流xxjl，"自信"程度越高，越希望召集会议
        else:
            if soc_net.power[agent_no][agent_no]['weight'] > uniform(0, 1):
                meet_info = {"type": "req", "name": "xxjl"}
        return agent, meet_info
    elif use_police == "jhjc_tljc":
        if 'tljc' in meet_req:
            max_relat = max([soc_net.relat[x][agent_no]['weight'] for x in meet_req['tljc']])
            if (max_relat > uniform(0, 1)):
                return agent, {"type": "commit", "name": "tljc"}
        last_agent = deepcopy(agent)
        agent = acts.act_jhnd(env, agent, Ti, Tfi)
        if (last_agent.a_plan is None or last_agent.a_plan.goal_value < agent.a_plan.goal_value):
            if soc_net.power[agent_no][agent_no]['weight'] > uniform(0, 1):
                meet_info = {"type": "req", "name": "xtfg"}
        return agent, meet_info
    elif use_police == "xdzx_xtfg":
        if 'xtfg' in meet_req:
            max_relat = max([soc_net.relat[x][agent_no]['weight'] for x in meet_req['xtfg']])
            if (max_relat > uniform(0, 1)):
                return agent, {"type": "commit", "name": "xtfg"}
        last_agent = deepcopy(agent)
        agent = acts.act_xdzx(env, agent, Ti, Tfi)
        # 如果计划执行完了(没计划)，或新的计划比原来好，召集讨论决策？
        if ((last_agent.a_plan is None) or (
                not agent.a_plan is None and last_agent.a_plan.goal_value < agent.a_plan.goal_value)):
            if soc_net.power[agent_no][agent_no]['weight'] > uniform(0, 1):
                meet_info = {"type": "req", "name": "tljc"}
        return agent, meet_info


def mul_agent_meet():
    pass


using_brain = [hqxx_zyzx, hqxx_zyzx, hqxx_zyzx, hqxx_zyzx, hqxx_zyzx]

act_brain = [mul_agent_act] * global_arg['Nagent']
meet_brain = [mul_agent_meet] * global_arg['Nagent']
