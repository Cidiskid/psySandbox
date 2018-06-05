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
        "Ts": 16,  # 每个stage的帧数
        "Nagent": 5  # Agent数量
    }
    return arg


def init_env_arg(global_arg):
    # NK model
    arg = {
        'N': 8,
        'K': 2,
        'P': 4,  # 每个位点状态by Cid
        'T': global_arg['T'],  # 模拟总时间
        'Tp': global_arg['T']  # 每个地形持续时间/地形变化耗时 by Cid
    }

    # 环境情景模型模块
    arg['ESM'] = {
        "f-req": 0.75,  # 适应要求，及格线
        "p-cplx": 1 - 0.75 / (1 + exp(arg['K'] - 5)),
        # (lambda Tp: 1 - 0.75 ** (1.0 * global_arg['T'] / Tp) / (1 + exp(arg['K'] - 5))),
        # TODO 修改公式
        "p-ugt": (1 - tanh(0.1 * (global_arg['Ts'] - 32))) * 0.5
    }

    # 区域相关参数，代表目标
    arg['area'] = {
        "sample_num": 100,
        "max_dist": 3,
        "mask_num": min(5, arg['N'])
    }

    # 个体可以采取的各项行动，行动本身的参数
    arg['ACT'] = {
        # 行动执行相关参数表
        'xdzx': {
            'kT0': 0.01,  #  default 0.5
            'cool_down': 0.995  # default 0.99
        },
        # 获取信息相关参数表
        'hqxx': {
            "mask_n": 4,    # 区域内的点和中心点有差异的位点数量
            "dist": 3,      # 区域半径，所有点和中心点的最大距离
            "dfs_p": 0.5,   # 表示多大概率往深了走
            "sample_n": 50  # 从区域中抽样的数量
        },
        # 计划拟定相关参数表
        'jhnd': {
            "sample_num": 50,
            "dfs_r": 0.5
        },
        # 计划决策相关参数表
        'jhjc': {
            "plan_eval": (lambda aim_value, lenght: aim_value * 0.99 * (len(lenght)))
        }
    }
    return arg


def init_agent_arg(global_arg, env_arg):
    arg = {}
    # 个体属性差异
    arg['a'] = {
        "insight": clip_rsmp(0.001, 0.999, Norm, mu=0.5, sigma=0.2),  # 环境感知能力
        "act": Norm(0.5, 0.1),  # 行动意愿
        "xplr": Norm(0.5, 0.3),  # 探索倾向
        "xplt": Norm(0.5, 0.3),  # 利用倾向
        "rmb": 8
    }

    # 适应分数观察值的偏差
    ob_a = 0.01  # default 0.025
    arg["ob"] = (lambda x: Norm(x, ob_a / arg['a']['insight']))  #原公式，
#    arg["ob"] = (lambda x: Norm(x, 0.05))  #测试公式

    arg['default'] = {
        "stage": {},  # 各种第0个stage的参数放在这里
        "frame": {  # 各种第0帧的参数放在这里
            # 主观情景模型
            "PSM": {
                "m-info": [],  # 新内容，存储各种临时信息
                "m-plan": [],
                "a-plan": None,
                'a-need': 0,  # 行动需求，原来的f-need
                's-sc': 0
            },
            # 行动偏好参数
            'ACT': {  # TODO 提问：和Brain的关系？
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

# 每帧刷新的参数列表
def init_frame_arg(global_arg, env_arg, agent_arg, stage_arg, last_arg, Tp, PSMfi):
    arg = {}

    arg['PSM'] = {
        "f-req": Norm(env_arg['ESM']['f-req'], 0.01 / agent_arg['a']['insight']),  # TODO 提问：只是初始值这样获得，是否移动到agent[default][frame]里？
        "p-cplx": Norm(env_arg['ESM']['p-cplx'], 0.01 / agent_arg['a']['insight']),  # 只是初始值这样获得
        "p-ugt": Norm(env_arg['ESM']['p-ugt'], 0.01 / agent_arg['a']['insight']),  # 只是初始值这样获得
        "m-info": deepcopy(last_arg['PSM']['m-info']),  # 新版用法不一样
        "m-plan": deepcopy(last_arg['PSM']['m-plan']),  # 新版用法不一样
        "a-plan": deepcopy(last_arg['PSM']['a-plan']),  # 拍死他丫的
        "s-sc": deepcopy(last_arg['PSM']['s-sc'])  # 新版用法不一样
    }

    # 计算当前个体在这一帧感知到的行动需求
    PSManeed_r = 1.0 / (1 + exp(5 * (PSMfi / arg['PSM']['f-req'] - 1)))
    PSManeed_a = 0.5
    arg['PSM']['a-need'] = PSManeed_a * last_arg['PSM']['a-need'] + (1 - PSManeed_a) * PSManeed_r

    # 判断当前个体在这一帧是否采取行动
    f1 = 1 + 0.5 * tanh(5 * (arg['PSM']['a-need'] - 0.75)) \
         + 0.5 * tanh(5 * (agent_arg['a']['act'] - 0.5))
    g1 = 1 - 0.2 * tanh(5 * (arg['PSM']['p-cplx'] - 0.625))
    h1 = 1 + 0.1 * cos(pi * (arg['PSM']['p-ugt'] - 0.5))
    arg['PROC'] = {
        'a-m': f1 * g1 * h1,  # 行动动机，代表行动意愿的强度
        'a-th': 0  # 行动阈值，初始0.6，测试版保证行动
    }
    arg['PROC']['action'] = (Norm(arg['PROC']['a-m'] - arg['PROC']['a-th'], 0.1) > 0)  # TRUE行动，FALSE不行动

# 以下参与用于确定采取何种行动的过程 TODO 提问：和Brain的关系？
    arg['ACT'] = {'p':{}}
    # 行动执行的偏好分
    xdzx_a = 0.5
    arg['ACT']['p']['xdzx'] = xdzx_a * last_arg['ACT']['p']['xdzx'] + (1 - xdzx_a) * 0.5  # 行动执行的偏好是常数，为0.5

    # 获取信息的偏好分
    hqxx_a = 0.5
    f2 = 1 - tanh(10 * (last_arg['PSM']['s-sc'] - 0.8 * arg['PSM']['f-req']))
    g2 = 1 + 0.2 * tanh(5 * (agent_arg['a']['xplr'] - 0.5))
    h2 = 1 + 0.1 * cos(pi * (arg['PSM']["p-ugt"] - 0.5))
    l2 = 1 + 0.2 * tanh(5 * (arg['PSM']['p-cplx'] - 0.5))
    arg['ACT']['p']['hqxx'] = hqxx_a * last_arg['ACT']['p']['hqxx'] + (1 - hqxx_a) * f2 * g2 * h2 * l2

    # 计划拟定的偏好分
    f3 = 1 + tanh(5 * (last_arg['PSM']['s-sc'] - 0.8 * arg['PSM']['f-req']))
    g3 = 1 + 0.2 * tanh(5 * (agent_arg['a']['xplt'] - 0.5))
    h3 = 2 + tanh(5 * (arg["PSM"]['p-ugt'] - 1))
    jhnd_a = 0.5
    arg['ACT']['p']['jhnd'] = jhnd_a * last_arg['ACT']['p']['jhnd'] + (1 - jhnd_a) * f3 * g3 * h3
    if (len(arg['PSM']['m-plan']) < 1):  # TODO 提问：函数的意思是？好像是老版本定义，需要修改。如果没有当前计划，计划拟定的权重为0
        arg['ACT']['p']['jhnd'] = 0

    arg['ACT']['choice'] = random_choice(softmaxM1(arg['ACT']['p']))

    return arg
