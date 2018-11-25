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


def agent_minfo(agents,env, **kargs):
    assert isinstance(agents, list)
    assert isinstance(env, Env)
    ret_result = {'each': [], 'common': {}}
    for agent in agents:
        assert isinstance(agent, Agent)

        def _all_area():
            return len([minfo for minfo in agent.frame_arg["PSM"]['m-info'] if minfo.dist > 0])

        def _unique_area(dist_thr):
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

        def _common_area(dist_thr):
            pass
        #TODO 通过排序来做，需要添加__gt__

        def _useful_m_info(dist_thr):
            useful_num = 0
            for area in agent.frame_arg['PSM']['m-info']:
                if area.dist < dist_thr:
                    continue
                if area.info['max'] > env.getValue(agent.state_now):
                    useful_num += 1
            return  useful_num

        metric = {
            "all_area": _all_area(),
            "unique_area": _unique_area(dist_thr=1),
            'unique_m_info': _unique_area(dist_thr=0),
            'useful_m_info': _useful_m_info(dist_thr=0)
        }
        ret_result['each'].append(metric)

    def sum_and_avg(key):
        res_sum = sum([m[key] for m in ret_result['each']])
        return {'sum': res_sum, "avg": res_sum/len(agents)}

    for key in ['all_area', 'unique_area', 'unique_m_info', 'useful_m_info']:
        ret_result['common'][key] = sum_and_avg(key)

    return ret_result


def register_all_metrics(the_metrics):
    assert isinstance(the_metrics, Metrics)
    the_metrics.add_metric(func=agent_minfo, path="agent.m-info", tags=['all', 'agent', 'stage'])
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