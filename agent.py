import logging
from copy import deepcopy
import arg
from env import NKmodel, State, Area

default_attrs = {}


class Agent:
    def __init__(self, arg):
        self.agent_arg = arg
        self.stage_arg = arg['default']['stage']
        self.frame_arg = arg['default']['frame']
        self.state_now = None
        self.inter_area = Area(State(0),
                               [True for i in range(State.N)],
                               (State.P // 2) * State.N)

    def RenewRsInfo(self, state, value, T):
        self.frame_arg["PSM"]['m-info'][int(state)] = {'T': T, 'value': value}
        rm_s = []
        for s in self.frame_arg["PSM"]['m-info']:
            if (self.frame_arg["PSM"]['m-info'][s]['T'] < T - self.agent_arg['a']['rmb']):
                rm_s.append(s)
        for s in rm_s:
            del self.frame_arg["PSM"]['m-info'][s]

    def renew_m_info_list(self, area_list, T):
        self.frame_arg["PSM"]['m-info'] += area_list
        for i in range(len(self.frame_arg["PSM"]['m-info']), -1, -1):
            if (self.frame_arg["PSM"]['m-info'][i].info['T_stmp'] < T - self.agent_arg['a']['rmb']):
                del self.frame_arg["PSM"]['m-info'][i]

    def renew_m_info(self, area, T):
        self.renew_m_info([area], T)

    def get_latest_m_info(self, latest_t, max_num):
        ret = [ar for ar in self.frame_arg['PSM']['m-info'] if ar.info['T_stmp'] >= latest_t]
        ret.sort(key=lambda ar: -ar.info['max'])
        return ret[:max_num]
