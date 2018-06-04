# -*- coding:utf-8 -*-
import env
import agent
import arg
import acts
from util import config,moniter,util
import logging

global_arg = arg.init_global_arg()

def xdzx_hqxx(env, agent, Ti, Tfi, agent_no):
    if (agent.frame_arg['PROC']['action']):
        if (env.getValue(agent.state_now, Ti) >= agent.inter_area.info['max'] * 0.99):
            logging.debug("Agent %d, act_hqxx" % (agent_no))
            agent = acts.act_hqxx(env, agent, Ti, Tfi)
        else:
            logging.debug("Agent %d, act_xdzx" % (agent_no))
            agent = acts.act_xdzx(env, agent, Ti, Tfi)
    return agent

def only_xdzx(env, agent, Ti, Tfi, agent_no):
    if (agent.frame_arg['PROC']['action']):
        logging.debug("Agent %d, act_xdzx" % (agent_no))
        agent = acts.act_xdzx(env, agent, Ti, Tfi)
    return agent

def single_ver1(env, agent, Ti, Tfi, agent_no):
    pass

using_brain = [only_xdzx, only_xdzx, only_xdzx, only_xdzx, only_xdzx]