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


def mul_agent_act(env, soc_net, agent, Ti, Tfi, agent_no, meet_req):
    assert isinstance(env, Env) and isinstance(soc_net, SoclNet) and isinstance(agent, Agent)
    prob = {"hqxx_xxjl": 1, "jhjc_tljc": 1, "xdzx_xtfg": 1}
    use_police = util.random_choice(prob)
    meet_info = None
    #    logging.debug("meet_req:{}".format(meet_req))
    #    logging.debug("agent_no:{}".format(agent_no))
    if use_police == "hqxx_xxjl":
        if 'xxjl' in meet_req:
            max_relat = max([soc_net.relat[x][agent_no]['weight'] for x in meet_req['xxjl']])
            if (max_relat > uniform(0, 1)):
                return agent, {"type": "commit", "name": "xxjl"}
        last_agent = deepcopy(agent)
        agent = acts.act_hqxx(env, agent, Ti, Tfi)
        if (last_agent.get_max_area().info['max'] < agent.get_max_area().info['max']):
            if soc_net.power[agent_no][agent_no]['weight'] > uniform(0, 1):
                meet_info = {"type": "req", "name": "tljc"}
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
