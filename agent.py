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
                               State.P * State.N)

    def RenewRsInfo(self, state, value, T):
        self.frame_arg["PSM"]['m-info'][int(state)] = {'T': T, 'value': value}
        rm_s = []
        for s in self.frame_arg["PSM"]['m-info']:
            if (self.frame_arg["PSM"]['m-info'][s]['T'] < T - self.agent_arg['a']['rmb']):
                rm_s.append(s)
        for s in rm_s:
            del self.frame_arg["PSM"]['m-info'][s]
