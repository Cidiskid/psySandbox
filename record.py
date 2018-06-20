# -*- coding:utf-8 -*-

import arg
import logging
from util.config import all_config
from util import moniter
from agent import Agent, Plan
from env import Env
from group import SoclNet
from functools import reduce
from collections import OrderedDict as ordict
from copy import deepcopy

class Record:
    def __init__(self):
        self.arg = arg.init_global_arg()
        self.T = self.arg['T']
        self.Ts = self.arg['Ts']
        self.Nagent = self.arg['Nagent']
        self.agents = [[]] * self.Nagent
        self.env = []
        self.socl_net = []

    @staticmethod
    def get_env_info(env):
        assert isinstance(env, Env)
        distri = env.getModelDistri()
        info = ordict()
        info['max'] = distri['max']
        info['avg'] = distri['avg']
        info['min'] = distri['min']
        return info

    @staticmethod
    def get_socl_net_info(socl_net):
        assert isinstance(socl_net, SoclNet)
        info = ordict()
        info['net'] = socl_net
        return info

    @staticmethod
    def get_agent_info(env, agent):
        assert isinstance(env, Env) and isinstance(agent, Agent)
        info = ordict()
        info['value'] = env.getValue(agent.state_now)
        if agent.a_plan is None:
            info['plan_goal'] = None
        else:
            info['plan_goal'] = agent.a_plan.goal_value
        return info

    # 尝试不调用getAllValue
    def add_env_record(self, env, T,up_info = None):
        assert isinstance(env, Env)
        if up_info is None:
            info = Record.get_env_info(env)
            # logging.info(info)
        else:
            info = ordict()
            info['max'] = up_info['nkinfo']['max']
            info['avg'] = up_info['nkinfo']['avg']
            info['min'] = up_info['nkinfo']['min']
            # logging.info(info)
        self.env.append((T,info))
        return info

    def add_socl_net_record(self, socl_net, T):
        assert isinstance(socl_net, SoclNet)
        info = Record.get_socl_net_info(socl_net)
        self.socl_net.append((T,info))
        return info

    def add_agent_record(self, env, agent, agent_no, T):
        assert isinstance(env, Env) and isinstance(agent, Agent)
        info = Record.get_agent_info(env, agent)
        self.agents[agent_no].append((T,info))
        return info

    def add_agents_record(self, env, agents, T):
        assert isinstance(env, Env) and len(agents) == self.Nagent
        assert reduce(lambda x, y: x and y, [isinstance(a, Agent) for a in agents])
        info = []
        for i in range(self.Nagent):
            info.append(self.add_agent_record(env, agents[i], i, T))
        return info
    def get_agent_record(self, agent_no, T):
        assert 0 <= agent_no < self.Nagent
        ret_t = 0
        for i in range(len(self.agents[agent_no])):
            if T < self.agents[agent_no][i][0]:
                return self.agents[agent_no][ret_t][1]
            ret_t = i
        return  self.agents[agent_no][ret_t][1]

    def output_socl_net_per_frame(self, T):
        title = ['frame', "from", "to", "power", "relat"]
        output_data = []
        for i in range(len(self.socl_net)-1, -1, -1):
            if(T >= self.socl_net[i][0]):
                for fr,to in self.socl_net[i][1]['net'].power.edges:
                    pow_w = self.socl_net[i][1]['net'].power[fr][to]['weight']
                    if (fr,to) in self.socl_net[i][1]['net'].relat.edges:
                        rel_w = self.socl_net[i][1]['net'].relat[fr][to]['weight']
                    else:
                        rel_w = None
                    output_data.append([T, fr, to, pow_w, rel_w])
                break
        return title, output_data
    def output_per_frame(self):
        pass

    def output_per_stage(self):
        pass
