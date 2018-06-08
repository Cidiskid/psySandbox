# -*- coding:utf-8 -*-
from group import Group, SoclNet
from util.config import all_config
import arg
import logging


def meeting_xtfg(env, agents, member, host, socl_net, T, Tfi):  # 协调分工
    pass


def meeting_xxjl(env, agents, member, host, socl_net, T, Tfi):  # 信息交流
    assert isinstance(host, set) and isinstance(member, set)
    assert host.issubset(member)
    ret_info = []
    for x in member:
        ret_info += agents[x].get_latest_m_info(env.arg['meeting']['xxjl']['last_p_t'],
                                                env.arg['meeting']['xxjl']['max_num'])
    for x in member:
        agents[x].renew_m_info_list(ret_info, T + Tfi)
    return agents, socl_net


def meeting_tljc():  # 讨论决策
    pass
