# -*- coding:utf-8 -*-
import logging
from copy import deepcopy
from env import State, Area, Env, get_area_sample_distr

default_attrs = {}


class Plan:
    def __init__(self, goal, goal_value, area):
        assert isinstance(goal, State) and isinstance(area, Area)
        assert area.state_in(goal)
        self.goal = goal
        self.goal_value = goal_value
        self.area = area
        self.info = {}

    def is_arrive(self, state):
        assert isinstance(state, State)
        return int(state) == int(self.goal)

    @staticmethod
    def _next_step(state_from, state_to):
        for i in range(State.N):
            if state_from[i] == state_to[i]:
                continue
            up_step = abs(state_to[i] - state_from[i] + State.P) % State.P
            dw_step = abs(state_from[i] - state_to[i] + State.P) % State.P
            if up_step <= dw_step:
                return state_from.walk(i, 1)
            else:
                return state_from.walk(i, -1)

    def next_step(self, state):
        if self.is_arrive(state):
            return state
        if self.area.state_in(state):
            return Plan._next_step(state, self.goal)
        else:
            return Plan._next_step(state, self.area.center)

    def len_to_finish(self, state):
        # 如果在区域内，计算点到目标到距离
        if self.area.state_in(state):
            return State.getDist(state, self.goal)
        # 否则计算点到区域中心+区域中心到goal的距离
        else:
            return State.getDist(state, self.area.center) \
                   + State.getDist(self.area.center, self.goal)


class Agent:
    def __init__(self, arg, env):
        assert isinstance(env, Env)
        self.agent_arg = arg
        self.stage_arg = arg['default']['stage']
        self.frame_arg = arg['default']['frame']
        self.state_now = None
        self.policy_now = None
        self.meeting_now = None
        self.a_plan = None

        # 添加一个全局area
        # start_area = Area(State(0), [True] * State.N, (State.P // 2) * State.N)
        # start_area.info = get_area_sample_distr(env=env,
        #                                        area=start_area,
        #                                        sample_num=env.arg['ACT']['hqxx']['sample_n'],
        #                                        T_stmp=0)


        # for k in start_area.info:
        #    start_area.info[k] = self.agent_arg['ob'](start_area.info[k])
        #self.frame_arg['PSM']['m-info'].append(start_area)


    def ob(self, env, state):
        assert isinstance(env, Env) and isinstance(state, State)
        return self.agent_arg['ob'](env.getValue(state))

    def renew_m_info_list(self, area_list, tfi):
        # 加入新信息
        self.frame_arg["PSM"]['m-info'] += area_list
        # 清空旧的信息
        for i in range(len(self.frame_arg["PSM"]['m-info']) - 1, -1, -1):
            if self.frame_arg["PSM"]['m-info'][i].info['T_stmp'] < tfi - self.agent_arg['a']['rmb']:
                del self.frame_arg["PSM"]['m-info'][i]

    def renew_m_info(self, area, tfi):
        self.renew_m_info_list([area], tfi)

    def get_latest_m_info(self, latest_t, max_num):
        ret = [ar for ar in self.frame_arg['PSM']['m-info'] if ar.info['T_stmp'] >= latest_t]
        ret.sort(key=lambda ar: -ar.info['max'])
        return ret[:max_num]

    def get_max_area(self):
        assert len(self.frame_arg['PSM']['m-info']) > 0
        max_area = self.frame_arg['PSM']['m-info'][0]
        for area in self.frame_arg['PSM']['m-info']:
            if max_area.info['max'] <= area.info['max']:
                max_area = area
        return max_area
