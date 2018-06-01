# -*- coding:utf-8 -*-
import os
from random import normalvariate as Norm
from random import uniform
from math import exp, pow, tanh, cos, pi
from copy import deepcopy
import logging
from util.util import max_choice, random_choice, softmaxM1, clip, clip_rsmp, clip_tanh
from env import Area

def init_global_arg():
    arg = {
        'T': 512,  # 模拟总时间
        "Ts": 512,  # 每个stage的帧数
        "Nagent": 5  # Agent数量
    }
    return arg


def init_env_arg(global_arg):
    # NK model
    arg = {
        'N': 8,
        'K': 0,
        'P':5,#每个位点状态by Cid
        'T': global_arg['T'],  # 模拟总时间
        'Tp': global_arg['T']  #每个地形持续时间/地形变化耗时 by Cid
    }

    # 环境情景模型模块
    arg['ESM'] = {
        "f-req": 0.75,  # 适应要求，及格线
        "p-cplx": 1 - 0.75  / (1 + exp(arg['K'] - 5)),
        #(lambda Tp: 1 - 0.75 ** (1.0 * global_arg['T'] / Tp) / (1 + exp(arg['K'] - 5))),
        #TODO 修改公式
        "p-ugt": (1 - tanh(0.1 * (global_arg['Ts'] - 32))) * 0.5
    }
    return arg


def init_agent_arg(global_arg, env_arg):
    arg = {}
    arg['a'] = {
        "insight": clip_rsmp(0.001, 0.999, Norm, mu=0.5, sigma=0.2),  # 环境感知能力
        "act": Norm(0.5, 0.1),  # 行动意愿
        "xplr": Norm(0.5, 0.3),  # 探索倾向
        "xplt": Norm(0.5, 0.3),  # 利用倾向
        "rmb": 8
    }
    arg["ob"] = (lambda x: Norm(x, 0.01 / arg['a']['insight']))
    arg['default'] = {
        "stage": {},
        "frame": {
            "PSM": {
                "m-info": {},  # 新内容，存储各种临时信息
                "m-plan": [],
                'a-need': 0,  # 行动需求，原来的f-need
                's-sc': 0
            },
            'ACT': {
                'p': {
                    'xdzx': 1,  # 行动执行
                    'hqxx': 0,  # 获取信息
                    'jhnd': 0  # 计划拟定
                }
            }
        }
    }
    return arg


def init_stage_arg(global_arg, env_arg, agent_arg, last_arg, T):
    return {}


def init_frame_arg(global_arg, env_arg, agent_arg, stage_arg, last_arg, Tp, PSMfi):
    arg = {}
    arg['PSM'] = {
        "f-req": Norm(env_arg['ESM']['f-req'], 0.01 / agent_arg['a']['insight']), #只是初始值这样获得
        "p-cplx": Norm(env_arg['ESM']['p-cplx'], 0.01 / agent_arg['a']['insight']),#只是初始值这样获得
        "p-ugt": Norm(env_arg['ESM']['p-ugt'], 0.01 / agent_arg['a']['insight']),#只是初始值这样获得
        "m-info": deepcopy(last_arg['PSM']['m-info']),#新版用法不一样
        "m-plan": deepcopy(last_arg['PSM']['m-plan']),#新版用法不一样
        "s-sc": deepcopy(last_arg['PSM']['s-sc']) #新版用法不一样
    }
    PSManeed_r = 1.0 / (1 + exp(5 * (PSMfi / arg['PSM']['f-req'] - 1)))
    PSManeed_a = 0.5
    arg['PSM']['a-need'] = PSManeed_a * last_arg['PSM']['a-need'] + (1 - PSManeed_a) * PSManeed_r

    arg['area'] = {
        "sample_num": 100,
        "max_dist": 3,
        "mask_num": min(5, env_arg['N'])
    }

    f1 = 1 + 0.5 * tanh(5 * (arg['PSM']['a-need'] - 0.75)) \
         + 0.5 * tanh(5 * (agent_arg['a']['act'] - 0.5))
    g1 = 1 - 0.2 * tanh(5 * (arg['PSM']['p-cplx'] - 0.625))
    h1 = 1 + 0.1 * cos(pi * (arg['PSM']['p-ugt'] - 0.5))
    arg['PROC'] = {
        'a-m': f1 * g1 * h1,  # 行动动机，代表行动意愿的强度
        'a-th': 0  # 行动阈值，初始0.6，测试版保证行动
    }
    arg['PROC']['action'] = (Norm(arg['PROC']['a-m'] - arg['PROC']['a-th'], 0.1) > 0)

    arg['ACT'] = {
        'p': {},
        'xdzx': {
            'kT0' : 0.5,
            'cool_down': 0.99
        },
        'hqxx': {
            'n_try': 7,
            'p_deep': 0.5
        },
        'jhnd': {}
    }
    xdzx_a = 0.5
    arg['ACT']['p']['xdzx'] = xdzx_a * last_arg['ACT']['p']['xdzx'] + (1 - xdzx_a) * 0.5  # 行动执行的偏好是常数，为0.5
    hqxx_a = 0.5
    f2 = 1 - tanh(10 * (last_arg['PSM']['s-sc'] - 0.8 * arg['PSM']['f-req']))
    g2 = 1 + 0.2 * tanh(5 * (agent_arg['a']['xplr'] - 0.5))
    h2 = 1 + 0.1 * cos(pi * (arg['PSM']["p-ugt"] - 0.5))
    l2 = 1 + 0.2 * tanh(5 * (arg['PSM']['p-cplx'] - 0.5))
    arg['ACT']['p']['hqxx'] = hqxx_a * last_arg['ACT']['p']['hqxx'] + (1 - hqxx_a) * f2 * g2 * h2 * l2
    f3 = 1 + tanh(5 * (last_arg['PSM']['s-sc'] - 0.8 * arg['PSM']['f-req']))
    g3 = 1 + 0.2 * tanh(5 * (agent_arg['a']['xplt'] - 0.5))
    h3 = 2 + tanh(5 * (arg["PSM"]['p-ugt'] - 1))
    jhnd_a = 0.5
    arg['ACT']['p']['jhnd'] = jhnd_a * last_arg['ACT']['p']['jhnd'] + (1 - jhnd_a) * f3 * g3 * h3
    if (len(arg['PSM']['m-plan']) < 1):
        arg['ACT']['p']['jhnd'] = 0

    arg['ACT']['choice'] = random_choice(softmaxM1(arg['ACT']['p']))

    return arg
