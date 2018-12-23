# -*- coding:utf-8 -*-
import arg
from util import config, moniter, util
from group import SoclNet
from env import Env
from agent import Agent
import logging
from copy import deepcopy
import pickle


class SingleMetric:
    def __init__(self, tags, func):
        self.tags = tags
        self.func = func

    def try_run(self, tags, **kargs):
        need_run = False
        for tag in tags:
            if tag in self.tags:
                need_run = True
        if need_run:
            return True, self.func(**kargs)
        else:
            return False, None


class Metrics:
    def __init__(self):
        self.tags = []
        self.data = {}
        self.metrics = {}

    def add_metric(self, func, path, tags):
        # m.add_metric(get_m_info. "agents.single.m_info", ["agent", "stage"])
        for tag in tags:
            if tag not in self.tags:
                self.tags.append(tag)
        point_data = self.data
        point_metric = self.metrics
        for key in path.split('.'):
            if key not in point_data:
                point_data[key] = {}
                point_metric[key] = {}
            point_data = point_data[key]
            point_metric = point_metric[key]
        point_data = {}
        point_metric['metric'] = SingleMetric(tags, func)
        point_data = None
        point_metric = None

    def calc_metric(self, tag_names, T, **kargs):
        # m.calc_metric(["stage"], agents=asdfasd, nets=asdfasdf, )
        for tag in tag_names:
            assert tag in self.tags, "{} not int matrics.tags".format(tag)
        logging.info("T: {}, tag_names: {}".format(T, tag_names))

        def _calc_metric(_metrics, _datas):
            for key in _metrics:
                if 'metric' in _metrics[key] and isinstance(_metrics[key]['metric'], SingleMetric):
                    is_run, result = _metrics[key]['metric'].try_run(tag_names, **kargs)
                    if is_run:
                        _datas[key][T] = result
                else:
                    _calc_metric(_metrics[key], _datas[key])

        _calc_metric(self.metrics, self.data)

    def get_data(self):
        # data的形式将会是： path（字典） + 一列表
        # mimimi = m['agents']['single'][1]
        # mimimi['m_info']
        # mimimi['uni_info']
        return self.data

    def save(self, path):
        with open(path, "w") as fp:
            pickle.dump(self.data, fp)


def agent_each_minfo(agents, env, **kargs):
    assert isinstance(agents, list)
    assert isinstance(env, Env)
    ret_result = {'each': [], 'avg': {}}

    def _all_area(agent):
        return len([minfo for minfo in agent.frame_arg["PSM"]['m-info'] if minfo.dist > 0])

    def _unique_area(agent, dist_thr):
        uniq_list = []
        for area in agent.frame_arg["PSM"]['m-info']:
            if area.dist >= dist_thr:
                flag = True
                for uq_item in uniq_list:
                    if uq_item == area:
                        flag = False
                        break
                if flag:
                    uniq_list.append(area)
        return len(uniq_list)

    def _useful_m_info(agent, dist_thr):
        useful_num = 0
        for area in agent.frame_arg['PSM']['m-info']:
            if area.dist < dist_thr:
                continue
            if area.info['max'] > env.getValue(agent.state_now):
                useful_num += 1
        return useful_num

    for agent in agents:
        assert isinstance(agent, Agent)
        metric = {
            "all_area": _all_area(agent),
            "unique_area": _unique_area(agent, dist_thr=1),
            'unique_m_info': _unique_area(agent, dist_thr=0),
            'useful_m_info': _useful_m_info(agent, dist_thr=0)
        }
        ret_result['each'].append(metric)

    def sum_and_avg(key):
        res_sum = sum([m[key] for m in ret_result['each']])
        return {'sum': res_sum, "avg": res_sum / len(agents)}

    for key in ['all_area', 'unique_area', 'unique_m_info', 'useful_m_info']:
        ret_result['avg'][key] = sum_and_avg(key)

    return ret_result


def agent_common_minfo(agents, env, **kargs):
    assert isinstance(agents, list)
    assert isinstance(env, Env)

    def _common_area(dist_thr):
        m_area_list = []
        for agent in agents:
            assert isinstance(agent, Agent)
            t_area = []
            for area in agent.frame_arg['PSM']['m-info']:
                if area.dist >= dist_thr:
                    t_area.append(area)
            t_area = sorted(t_area)
            for i in range(len(t_area) - 1, 0, -1):
                if t_area[i] == t_area[i - 1]:
                    del t_area[i]
            m_area_list += t_area
        m_area_list = sorted(m_area_list)
        same_cnt = 0
        same_cnt_list = []
        for i in range(len(m_area_list)):
            if i == 0 or m_area_list[i] == m_area_list[i - 1]:
                same_cnt += 1
            else:
                same_cnt_list.append(same_cnt)
                same_cnt = 1
        same_cnt_list.append(same_cnt)
        same_hist = [0] * (max(same_cnt_list) + 1)
        for x in same_cnt_list:
            same_hist[x] += 1
        return same_hist

    ret_result = {
        "common_area": _common_area(1),
        "common_m_info": _common_area(0),
    }
    return ret_result

def agent_each_mplan(agents, env, **kargs):
    assert isinstance(agents, list)
    assert isinstance(env, Env)
    ret_result = {'each': [], 'avg': {}}

    def _all_plan(agent):
        return len(agent.frame_arg["PSM"]['m-plan'])

    def _unique_plan(agent):
        uniq_list = deepcopy(agent.frame_arg["PSM"]['m-plan'])
        uniq_list = sorted(uniq_list)
        for i in range(len(uniq_list) - 1, 0, -1):
            if uniq_list[i] == uniq_list[i - 1]:
                del uniq_list[i]
        return len(uniq_list)

    for agent in agents:
        assert isinstance(agent, Agent)
        metric = {
            "all_plan": _all_plan(agent),
            "unique_plan": _unique_plan(agent),
        }
        ret_result['each'].append(metric)

    def sum_and_avg(key):
        res_sum = sum([m[key] for m in ret_result['each']])
        return {'sum': res_sum, "avg": res_sum / len(agents)}

    for key in ['all_plan', 'unique_plan']:
        ret_result['avg'][key] = sum_and_avg(key)

    return ret_result

def agent_common_mplan(agents, env, **kwargs):
    assert isinstance(agents, list)
    assert isinstance(env, Env)

    def _common_mplan():
        m_mplan_list = []
        for agent in agents:
            assert isinstance(agent, Agent)
            t_plan = deepcopy(agent.frame_arg['PSM']['m-plan'])
            t_plan = sorted(t_plan)
            for i in range(len(t_plan) - 1, 0, -1):
                if t_plan[i] == t_plan[i - 1]:
                    del t_plan[i]
            m_mplan_list += t_plan
        m_mplan_list = sorted(m_mplan_list)
        same_cnt = 0
        same_cnt_list = []
        for i in range(len(m_mplan_list)):
            if i == 0 or m_mplan_list[i] == m_mplan_list[i - 1]:
                same_cnt += 1
            else:
                same_cnt_list.append(same_cnt)
                same_cnt = 1
        same_cnt_list.append(same_cnt)
        same_hist = [0] * (max(same_cnt_list) + 1)
        for x in same_cnt_list:
            same_hist[x] += 1
        return same_hist

    def _working_plan():
        ret_result = {}
        a_plan_list = sorted([agent.a_plan for agent in agents if agent.a_plan is not None])
        ret_result['num'] = len(a_plan_list)
        same_cnt = 0
        same_cnt_p = []
        for i in range(len(a_plan_list)):
            if i == 0 or a_plan_list[i] == a_plan_list[i -1]:
                same_cnt += 1
            else:
                if same_cnt > 1:
                    same_cnt_p.append(same_cnt * 1.0 / len(a_plan_list))
                same_cnt = 1
        if same_cnt > 1:
            same_cnt_p.append(same_cnt * 1.0 / len(a_plan_list))
        ret_result['ratio'] = same_cnt_p
        return ret_result

    ret_result = {
        "common_plan": _common_mplan(),
        "working_plan": _working_plan()
    }
    return ret_result

#def act_leadership

def agent_network(agents, socl_net, **kwargs):
    assert isinstance(socl_net, SoclNet)
    return {
        'relate_Cc': socl_net.get_power_out_close_centrality(),
        'power_Cod': socl_net.get_relat_close_centrality()
    }

def register_all_metrics(the_metrics):
    assert isinstance(the_metrics, Metrics)
    the_metrics.add_metric(func=agent_each_minfo, path="agent.m-info.each", tags=['all', 'agent', 'frame'])
    the_metrics.add_metric(func=agent_common_minfo, path="agent.m-info.common", tags=['all', 'agent', 'frame'])
    the_metrics.add_metric(func=agent_each_mplan, path="agent.m-plan.each", tags=['all', 'agent', 'frame'])
    the_metrics.add_metric(func=agent_common_mplan, path="agent.m-plan.common", tags=['all', 'agent', 'frame'])
    the_metrics.add_metric(func=agent_network, path="agent.network", tags=['all', 'agent', 'network', 'stage'])
    return the_metrics


if __name__ == '__main__':
    moniter.LogInit()
    mer = Metrics()


    def nums_avgs(nums, **kargs):
        return sum(nums) / len(nums)


    def nums_lens(nums, **kargs):
        return len(nums)


    def strs_nos_num(strs, **kargs):
        return len([s for s in strs if 's' not in s])


    mer.add_metric(func=nums_avgs, path="num.avg", tags=['all', 'num'])
    mer.add_metric(func=nums_lens, path="num.len", tags=['all', 'num'])
    mer.add_metric(func=strs_nos_num, path="str.noS", tags=['all', 'str'])

    from random import randint

    nums = [randint(0, 10) for i in range(10)]

    for i in range(10):
        if i % 2 == 0:
            mer.calc_metric(['str'], i, nums=nums, strs=[])
        mer.calc_metric(['num'], i, nums=nums)
        nums.append(randint(0, 10))
    import json

    print(json.dumps(mer.data, indent=2))
