# -*- coding:utf-8 -*-
import os
from random import normalvariate as Norm
from random import uniform, paretovariate, random
from math import exp, pow, tanh, cos, pi
from copy import deepcopy
import logging
from util.util import max_choice, random_choice, softmaxM1, clip, clip_rsmp, clip_tanh, rand_shrink
from env import Area


def init_global_arg():
    arg = {
        'T': 256,  # default1224 256 模拟总时间
        "Ts": 8,  # default1224 8 每个stage的帧数
        "Nagent": 16,  # default1224 16 Agent数量
        'D_env': True,  # 动态地形开关
        'mul_agent': True,  # 多人互动开关
        'repeat': 1  # 重复几次同样参数的实验
    }
    return arg


def init_env_arg(global_arg):
    # NK model
    arg = {
        'N': 5,  # default1224 5
        'K': 3,  # default1224 3
        'P': 7,  # # default1224 7 每个位点状态
        'T': global_arg['T'],  # 模拟总时间
        'Tp': global_arg['T'],  # 每个地形持续时间/地形变化耗时 by Cid
        'dynamic': global_arg['D_env'],  # 动态地形开关
        'sigma': 0.095,  # default1224 0.095
        'mu': 0.5,  # default1224 0.5
        'value2ret': (lambda real_value: min(max(0.5 + 0.5 * (real_value - arg['mu']) / (4 * arg['sigma']), 0), 1))
        # default1224 99%截断，并将区间调整为[0，1]
        # 'value2ret': (lambda real_value: real_value)  # 原始值
    }

    # 环境情景模型模块 已弃用
    arg['ESM'] = {
        "f-req": 0.75,  # 适应要求，及格线
        "p-cplx": 1 - 0.75 / (1 + exp(arg['K'] - 5)),
        # (lambda Tp: 1 - 0.75 ** (1.0 * global_arg['T'] / Tp) / (1 + exp(arg['K'] - 5))),

        "p-ugt": (1 - tanh(0.1 * (global_arg['Ts'] - 32))) * 0.5
    }

    # 区域相关参数，代表目标 已经无效
    # arg['area'] = {
    #    "sample_num": 50,  # 抽样个数
    #    "max_dist": 3,  # 游走距离
    #    "mask_num": min(5, arg['N'])  # 可移动的位点限制
    # }

    plan_a = 0.1  # default7.4 0.1 距离对计划得分影响系数
    arg['plan'] = {
        # 计划得分
        'eval': (lambda dist, trgt: trgt * (1 - plan_a * (1 - trgt)) ** dist)
    }

    # 个体可以采取的各项行动，行动本身的参数
    arg['ACT'] = {
        # 行动执行相关参数表
        'zyzx': {
            'ob': (lambda x: Norm(x, 0.2))  # 自由执行的ob误差为所有agent相同的固定值 default7.4 0.2
            # zyzx自由执行相关参数
        },
        'xdzx': {
            # 执行计划的概率
            'do_plan_p': (
                lambda st_val, dist, trgt: 0.5 + 0.5 * tanh(10 * (arg['plan']['eval'](dist, trgt) - st_val))),
            'kT0': 0.05,  # default7.4 0.05
            'cool_down': 0.95,  # default7.4 0.95
        },
        # 获取信息相关参数表
        'hqxx': {
            "mask_n": 2,  # default7.4 2 区域的方向夹角大小，指区域内的点中允许变化的位点数量
            "dist": 6,  # default7.4 6 区域半径，所有点和中心点的最大距离
            "dfs_p": 0.5,  # default7.4 0.5 表示多大概率往深了走
            "sample_n": 20  # default7.4 30 从区域中抽样的数量
        },
        # 计划拟定相关参数表
        'jhnd': {
            "sample_num": 8,
            "dfs_r": 0.5
        },
        # 计划决策相关参数表
        'jhjc': {
            "plan_eval": arg['plan']['eval']
        },
        "whlj": {
            "k": global_arg['Nagent'] // 2,
            "delta_relate": lambda old: 0.1  # 改变速率default7.4 0.1，可以手动修改
        },
        "dyjs": {
            "delta_power": lambda old: 0.05  # 改变速率default7.4 0.05，可以手动修改
        }
    }

    # 集体行动相关参数表
    arg['meeting'] = {
        'xxjl': {
            'last_p_t': 32,  # default7.4 32 最近多少帧内的信息
            'max_num': 3  # default7.4 3 最多共享多少个
        }
    }
    return arg


# 社会网络相关参数
def init_soclnet_arg(global_arg, env_arg):
    arg = {}
    arg['Nagent'] = global_arg['Nagent']
    arg['power_thld'] = 0.7  # default7.4 0.7  表示可以被强化角色定义的最小影响力分数
    arg['relat_thld'] = 0.7  # default7.4 0.7
    arg['relat_init'] = 0.3  # default7.4 0.5
    arg['relat_turb'] = 0.15  # default7.4 0.15
    arg['power_init'] = 0.3  # default7.4 0.5
    arg['power_turb'] = 0.15  # default7.4 0.15

    # 权重到距离的转化公式
    arg['pow_w2d'] = (lambda x: 1 / (0.01 + x) + 0.01)  # default7.4

    arg['re_decr_r'] = 0.95  # default7.4 0.95 自然衰减率

    return arg


def init_agent_arg(global_arg, env_arg):
    arg = {}
    # 个体属性差异
    arg['a'] = {
        #"insight": clip_rsmp(0.001, 9.999, paretovariate, alpha=1) / 10,  # 环境感知能力 base 模式
        # "insight": clip_rsmp(0.50, 0.85, uniform, a=0.50, b=0.85), # expert模式
        "insight": random(), # 随机模式
        "act": clip_rsmp(-0.999, 0.999, Norm, mu=0, sigma=0.1),  # default7.4  Norm(0, 0.1),  # 行动意愿
        #"xplr": clip_rsmp(-0.999, 0.999, Norm, mu=0, sigma=0.35),  # default7.4  Norm(0, 0.3),  # 探索倾向
        #"xplr": clip_rsmp(-0.999, 0.999, Norm, mu=0.3, sigma=0.35),  # default7.4  Norm(0, 0.3),  # 高探索倾向
        "xplr": clip_rsmp(-0.999, 0.999, Norm, mu=0, sigma=0.6),  # default7.4  Norm(0, 0.3),  # 高探索异质性
        #"xplt": clip_rsmp(-0.999, 0.999, Norm, mu=0, sigma=0.35),  # default7.4  Norm(0, 0.3),  # 利用倾向
        "xplt": clip_rsmp(-0.999, 0.999, Norm, mu=0, sigma=0.6),  # default7.4  Norm(0, 0.3),  # 高利用异质性
        #"xplt": clip_rsmp(-0.999, 0.999, Norm, mu=0.3, sigma=0.35),  # default7.4  Norm(0, 0.3),  # 高利用倾向
        "enable": clip_rsmp(-0.999, 0.999, Norm, mu=0, sigma=0.1),  # default7.4  Norm(0, 0.1),
        #"enable": clip_rsmp(-0.999, 0.999, Norm, mu=0.3, sigma=0.35),  # 高enable模式
        # "enable": clip_rsmp(-0.999, 0.999, uniform, a=-0.8, b=-0.2),  # 对事不对人（不管团队无enable）模式
        # "se": clip_rsmp(0.001, 0.999, Norm, mu=arg['a']["insight"], sigma=0.1)
        "m-info-rmb": 32,   # default1224 64
        "m-plan-rmb": 32,   # default1224 64
    }

    # 适应分数观察值的偏差
    ob_a = 0.25  # default7.4 0.25
    arg["ob"] = (lambda x: Norm(x, ob_a * (1 - arg['a']['insight'])))  # 更换为1-a_insight
    # arg["ob"] = (lambda x: x)  # 测试公式

    incr_rate = 0.15  # default7.4 0.15 关系增加速率
    arg['d_re_incr_g'] = (lambda old_re: incr_rate * (1 - 0.5 * old_re))  # 计算每次更新多少，可以随意动
    # default7.4 arg['d_re_incr_g'] = (lambda old_re: incr_rate * (1 - 0.5 * old_re))
    arg["re_incr_g"] = lambda old_re: \
        max(min(old_re + arg['d_re_incr_g'](old_re), 1), 0)  # 表示general的increase，在参加完任意一次集体活动后被调用

    arg['dP_r'] = {
        "other": 0.5,  # default7.4 0.5 对他人给的计划变化幅度更大
        "self": 0.1  # default7.4 0.1 对自己的计划变化幅度较小（效能提升小）
    }
    dP_s = 50  # default7.4 10 对变化的敏感度,值越大，敏感度越高
    # arg["dPower"] = (lambda dF, dP_r: dP_r * tanh(dP_s * dF))
    arg["dPower"] = (lambda dF, dP_r: dP_r * rand_shrink(tanh(dP_s * dF), 0.5))  # default7.4
    arg["pwr_updt_g"] = lambda old_pwr, dP: \
        max(min(((1 - 0.5 * abs(dP)) * old_pwr + 0.5 * (dP + abs(dP))), 1), 0)  # abs(dp)前添加0.5系数,加速变化，留下buffer
    # default7.4 max(min(((1 - 0.5 * abs(dP)) * old_pwr + 0.5 * (dP + abs(dP))), 1), 0)  # abs(dp)前添加0.5系数,加速变化，留下buffer
    arg["d_pwr_updt_g"] = (lambda old_pwr, dP: arg["pwr_updt_g"](old_pwr, dP) - old_pwr)

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


def init_group_arg(global_arg, env_arg, T):
    arg = {}
    return arg


def init_stage_arg(global_arg, env_arg, agent_arg, last_arg, T):
    return {}


# 每帧刷新的参数列表
def init_frame_arg(global_arg, env_arg, agent_arg, stage_arg, last_arg, Tp, PSMfi):
    arg = {}

    arg['PSM'] = {
        "f-req": Norm(env_arg['ESM']['f-req'], 0.01 / agent_arg['a']['insight']),
        "p-cplx": Norm(env_arg['ESM']['p-cplx'], 0.01 / agent_arg['a']['insight']),  # 只是初始值这样获得
        "p-ugt": Norm(env_arg['ESM']['p-ugt'], 0.01 / agent_arg['a']['insight']),  # 只是初始值这样获得
        "m-info": deepcopy(last_arg['PSM']['m-info']),  # 新版用法不一样
        "m-plan": deepcopy(last_arg['PSM']['m-plan']),  # 新版用法不一样
        "a-plan": deepcopy(last_arg['PSM']['a-plan']),  # 拍死他丫的
        "s-sc": deepcopy(last_arg['PSM']['s-sc'])  # 新版用法不一样
    }

    # 计算当前个体在这一帧感知到的行动需求
    # PSManeed_r = 1.0 / (1 + exp(5 * (PSMfi / arg['PSM']['f-req'] - 1)))
    # PSManeed_a = 0.5
    # arg['PSM']['a-need'] = PSManeed_a * last_arg['PSM']['a-need'] + (1 - PSManeed_a) * PSManeed_r

    # 判断当前个体在这一帧是否采取行动
    # f1 = 1 + 0.5 * tanh(5 * (arg['PSM']['a-need'] - 0.75)) \
    #    + 0.5 * tanh(5 * (agent_arg['a']['act'] - 0.5))
    # g1 = 1 - 0.2 * tanh(5 * (arg['PSM']['p-cplx'] - 0.625))
    # h1 = 1 + 0.1 * cos(pi * (arg['PSM']['p-ugt'] - 0.5))
    # arg['PROC'] = {
    #    'a-m': f1 * g1 * h1,  # 行动动机，代表行动意愿的强度
    #    'a-th': 0  # 行动阈值，初始0.6，测试版保证行动
    # }
    arg['PROC'] = {}
    arg['PROC']['action'] = True  # 目前始终行动
    # arg['PROC']['action'] = (Norm(arg['PROC']['a-m'] - arg['PROC']['a-th'], 0.1) > 0)  # TRUE行动，FALSE不行动

    # 行动执行的偏好基础参数
    xdzx_a = 2  # default7.4 0.5 行动执行偏好系数
    hqxx_a = 2  # default7.4 0
    jhjc_a = 1.5  # default7.4 0
    whlj_a = 1
    dyjs_a = 1
    tjzt_a = 0
    xdzx_ob = 1  # ob = odds base default7.4 1
    hqxx_ob = 1 + agent_arg['a']['xplr']
    jhjc_ob = 1 + agent_arg['a']['xplt']
    whlj_ob = 0.5 + agent_arg['a']['enable']
    dyjs_ob = 1 + agent_arg['a']['enable']
    tjzt_ob = 1 + agent_arg['a']['enable']
    xdzx_rp = 0.5  # 行动执行随pan变化的最大幅度,dplan一定>0 default7.4 0.5
    jhjc_ra = 0.5  # 计划决策随area变化的最大幅度 default7.4 0.5
    arg['ACT'] = {
        'odds': {
            "xdzx": lambda darea, dplan: -1 + 0.5 + exp(xdzx_a * xdzx_ob * (1 + xdzx_rp * tanh(50 * dplan))),
            "hqxx": lambda darea, dplan: -1 + exp(hqxx_a * hqxx_ob),
            "jhjc": lambda darea, dplan: -1 + exp(jhjc_a * jhjc_ob * (1 + jhjc_ra * tanh(50 * darea))),
            "whlj": lambda darea, dplan: -1 + exp(whlj_a * whlj_ob),
            "dyjs": lambda darea, dplan: -1 + exp(dyjs_a * dyjs_ob),
            "tjzt": lambda darea, dplan: 0  # 先去掉这个选项
        },
        "p": {},
        "p-cmt": {},
        "p-req": {}
    }
    k_cmt = 0
    amp_cmt = 1  # default7.4 1.5
    arg['ACT']['p-cmt']['xxjl'] = lambda max_relat, max_power: \
        min(amp_cmt * ((1 - k_cmt) * max_relat ** 2 + k_cmt), 1)
    arg['ACT']['p-cmt']['tljc'] = lambda max_relat, max_power: \
        min(amp_cmt * ((1 - max_power) * max_relat ** 2 + max_power), 1)
    arg['ACT']['p-cmt']['xtfg'] = lambda max_relat, max_power: \
        min(amp_cmt * ((1 - max_power) * max_relat ** 2 + max_power), 1)
    # (1 - max(0, max_power - self_efficacy)) * max_relat ** 2 + max(0, max_power - self_efficacy)  # 原公式
    k_req = 0
    amp_req = 1.2  # default7.4 1.5
    arg['ACT']['p-req']['xxjl'] = lambda host_Cc, host_Coc: \
        min(amp_req * ((1 - k_req) * host_Cc ** 2 + k_req), 1)
    arg['ACT']['p-req']['tljc'] = lambda host_Cc, host_Coc: \
        min(amp_req * ((1 - host_Coc) * host_Cc ** 2 + host_Coc), 1)  # default7.4
    # (1 - self_efficacy) * host_Cod ** 2 + self_efficacy
    arg['ACT']['p-req']['xtfg'] = lambda host_Cc, host_Coc: \
        min(amp_req * ((1 - host_Coc) * host_Cc ** 2 + host_Coc), 1)  # default7.4
    # (1 - self_efficacy) * host_Cod ** 2 + self_efficacy

    return arg
