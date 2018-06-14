# -*- coding:utf-8 -*-
from util.config import all_config
from util import moniter
import logging
import random
from copy import deepcopy
import arg


class State:
    N = 0
    P = 0

    def __init__(self, data=None):
        if (data is None):
            self.data = [0 for _ in range(State.N)]
        elif (type(data) == list):
            assert len(data) == State.N
            assert max(data) < State.P and min(data) >= 0
            self.data = deepcopy(data)
        elif (type(data) == int):
            assert 0 <= data < State.P ** State.N
            self.data = State.int2list(data)
        else:
            raise Exception("Init data must be int or list!")

    def __int__(self):
        return State.list2int(self.data)

    def __str__(self):
        if (State.P < 10):
            return ''.join([str(_) for _ in self.data])
        return '_'.join([str(_) for _ in self.data])

    def __getitem__(self, item):
        assert 0 <= item < self.N
        return self.data[item]

    def __setitem__(self, key, value):
        assert 0 <= key < self.N and 0 <= value < self.P
        self.data[key] = value

    def walk(self, key, d):
        assert 0 <= key < self.N
        to_ret = deepcopy(self)
        to_ret[key] = (to_ret[key] + d) % self.P
        return to_ret

    def walk_delta(self, delta):
        assert len(delta) == self.N
        to_ret = deepcopy(self)
        for i in range(len(delta)):
            to_ret[i] = (to_ret[i] + delta[i]) % self.P
        return to_ret

    @staticmethod
    def getGrayCode(N, no):
        def XOR(a, b):
            return int(a + b - 2 * a * b)

        ret_code = []
        for i in range(N):
            ret_code.append(no % 2)
            no = no // 2
        for i in range(N - 1):
            ret_code[i] = XOR(ret_code[i + 1], ret_code[i])
        return ret_code

    @staticmethod
    def int2list(num):
        code = []
        for i in range(State.N):
            code.append(num % State.P)
            num //= State.P
        return code

    @staticmethod
    def list2int(code):
        return int(sum([code[i] * (State.P ** int(i)) for i in range(len(code))]))

    @staticmethod
    def getDist(s1, s2):
        assert isinstance(s1, State) and isinstance(s2, State)

        def bitDist(a, b, p):
            return min((a - b + p) % p, (b - a + p) % p)

        return sum([bitDist(s1[i], s2[i], State.P) for i in range(State.N)])

    @staticmethod
    def getDiffFrom(s1, s2):
        assert isinstance(s1, State) and isinstance(s2, State)

        def bitDiffFrom(a, b, p):
            if (a < b):
                return b - a if (abs(b - a) <= abs(b - (a + p))) else b - (a + p)
            else:
                return b - a if (abs(b - a) <= abs((b + p) - a)) else (b + p) - a

        return [bitDiffFrom(s1[i], s2[i], State.P) for i in range(State.N)]


class Area:
    def __init__(self, center, mask, dist):
        assert isinstance(center, State) and len(mask) == center.N
        self.center = center
        self.mask = mask
        self.dist = dist
        self.info = {}

    def get_dist(self, state):
        return State.getDist(state, self.center)

    def get_mask_num(self):
        return sum([int(x) for x in self.mask if int(x) == 1])

    def getAllPoint(self):
        logging.debug("start")
        all_point = {int(self.center)}
        bfs_queue = [(self.center, self.dist)]
        head = 0
        while (head < len(bfs_queue)):
            s, deep = bfs_queue[head]
            head += 1
            if (deep <= 0):
                continue
            for i in range(len(self.mask)):
                if (int(self.mask[i]) == 1):
                    for dlt in [-1, 1]:
                        st = s.walk(i, dlt)
                        i_st = int(st)
                        if (not i_st in all_point):
                            all_point.add(i_st)
                            bfs_queue.append((st, deep - 1))
        logging.info(
            "Area:(c:%s, mask:%s, d:%d):num %d" % (str(self.center), str(self.mask), self.dist, len(bfs_queue)))
        return [pair[0] for pair in bfs_queue]

    def state_in(self, state):
        assert isinstance(state, State) and state.N == self.center.N
        diff = State.getDiffFrom(state, self.center)
        if (State.getDist(state, self.center) > self.dist):
            return False
        for i in range(state.N):
            if (int(self.mask[i]) == 0 and diff[i] != 0):
                return False
        return True

    # TODO 解释具体原理
    def rand_walk(self, state):
        assert isinstance(state, State)
        assert self.state_in(state)
        if (self.dist <= 0 or sum([int(_) for _ in self.mask]) <= 0):  # 若输入dist或mask小于等于零，无效参数，中断
            return state
        able_walk = [i for i in range(state.N) if (int(self.mask[i]) == 1)]  # 随机选择mask位允许修改
        while (True):
            state_t = state.walk(random.sample(able_walk, 1)[0],
                                 random.sample([-1, 1], 1)[0])
            if (self.state_in(state_t)):
                return state_t

    def sample_near(self, state, sample_num, dfs_r=0):
        logging.debug("start")
        assert isinstance(state, State)
        assert self.state_in(state)
        retry_num = sample_num
        try_queue = [state]
        head = 0
        while (retry_num > 0 and len(try_queue) < sample_num + 1):
            st = self.rand_walk(try_queue[head])
            if ((not st in try_queue) and self.state_in(st)):
                try_queue.append(st)
                if (random.uniform(0, 1) <= dfs_r and head < len(try_queue) - 1):
                    head += 1
                retry_num = sample_num
            else:
                retry_num -= 1
        return try_queue


class NKmodel:
    def __init__(self, n, k, p=2):
        assert (type(n) == int and type(k) == int and type(p) == int)
        assert (0 <= k < n)
        logging.debug("Init NK model n=%d,k=%d" % (n, k))
        self.N = n
        self.K = k
        self.P = p
        self.theta_func = NKmodel.genRandTheta(self.N, self.K, self.P)

    @staticmethod
    def genRandTheta(n, k, p):
        logging.debug("Init NK model -> genRandTheta")
        k_c_max = p ** (k + 1)
        logging.debug("genRandTheta: k_c_max %d*%d=%d" % (n, k_c_max, n * k_c_max))
        ret_map = []
        for i in range(n):
            ret_map.append([])
            for j in range(k_c_max):
                ret_map[i].append(random.random())
        return ret_map

    def getValue(self, state):
        assert (state.N == self.N and state.P == self.P)
        rtn_value = 0
        code_t = state.data + state.data
        for i in range(self.N):
            rtn_value += self.theta_func[i][State.list2int(code_t[i:i + self.K + 1])]
        return rtn_value / self.N


class Env:
    def __init__(self, arg):
        self.arg = arg
        self.N = arg['N']
        self.K = arg['K']
        self.P = arg['P']
        State.N = self.N
        State.P = self.P
        self.models = {"st": NKmodel(self.N, self.K, self.P),
                       "ed": NKmodel(self.N, self.K, self.P)}
        self.T = arg['T']
        self.T_clock = 0
        self.ESM = arg['ESM']

    def set_clock(self, T):
        self.T_clock = T

    # 动态过程的实现，通过getValue改变
    def getValue(self, state, t=None):
        assert (state.N == self.N and state.P == self.P)
        if (t is None):
            t = self.T_clock
        value_st = self.models["st"].getValue(state)
        value_ed = self.models["ed"].getValue(state)
        return value_st + (value_ed - value_st) * t / self.T

    def getAllValue(self):
        logging.debug("start")
        return [self.getValue(State(i)) for i in range(self.P ** self.N)]

    def getAllPeakValue(self):
        logging.debug("start")
        peak_value = []
        for i in range(self.P ** self.N):
            state = State(i)
            state_value = self.getValue(state)
            flag = False
            for j in range(self.N):
                for dl in [-1, 1]:
                    state_t = state.walk(j, dl)
                    if (state_value < self.getValue(state_t)):
                        flag = True
                        break
            if (not flag):
                peak_value.append(state_value)
        logging.info("Ti: {}, peak num: {}".format(self.T_clock, len(peak_value)))
        return peak_value

    @staticmethod
    def _getDistri(nums):
        all_value = sorted(nums)
        return {
            "max": all_value[-1],
            "min": all_value[0],
            "avg": sum(all_value) / len(all_value),
            "mid": all_value[len(all_value) // 2],
            "p0.16": all_value[int(round((len(all_value) - 1) * 0.16))],
            "p0.84": all_value[int(round((len(all_value) - 1) * 0.84))]
        }

    def getModelDistri(self):
        return Env._getDistri(self.getAllValue())

    def getModelPeakDistri(self):
        return Env._getDistri(self.getAllPeakValue())


def get_area_sample_value(env, area, sample_num, state=None, dfs_r=0.5):
    if (state is None):
        state = area.center
    states = area.sample_near(state, sample_num, dfs_r)
    return [env.getValue(s) for s in states]


def get_area_sample_distr(env, area, sample_num, T_stmp, state=None, dfs_r=0.5):
    if (state is None):
        state = area.center
    logging.debug("start")
    state_values = get_area_sample_value(env, area, sample_num, state, dfs_r)
    all_value = sorted(state_values)
    return {
        "max": all_value[-1],
        "min": all_value[0],
        "avg": sum(all_value) / len(all_value),
        "mid": all_value[len(all_value) // 2],
        "p0.16": all_value[int(round((len(all_value) - 1) * 0.16))],
        "p0.84": all_value[int(round((len(all_value) - 1) * 0.84))],
        'T_stmp': T_stmp
    }


if (__name__ == "__main__"):
    import numpy as np

    all_config.load()
    moniter.LogInit()
    logging.info("Start")
    global_arg = arg.init_global_arg()
    env_arg = arg.init_env_arg(global_arg)
    N = env_arg['N']
    k = env_arg['K']
    P = env_arg['P']
    T = env_arg['T']
    env = Env(env_arg)
    print(env.getModelPeakDistri())
