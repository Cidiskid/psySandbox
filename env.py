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


class Area:
    def __init__(self):
        pass


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
        self.ESM = arg['ESM']

    def getValue(self, state, t):
        assert (state.N == self.N and state.P == self.P)
        value_st = self.models["st"].getValue(state)
        value_ed = self.models["ed"].getValue(state)
        return value_st + (value_ed - value_st) * t / self.T

    def getAllValue(self, t):
        return [self.getValue(State(i), t) for i in range(self.P ** self.N)]

    def getAllPeakValue(self, t):
        peak_value = []
        for i in range(self.P ** self.N):
            state = State(i)
            state_value = self.getValue(state, t)
            flag = False
            for j in range(self.N * 2):
                state_t = state.walk(j // 2, 1 - 2 * (j % 2))
                if (state_value < self.getValue(state_t, t)):
                    flag = True
                    break
            if (not flag):
                peak_value.append(state_value)
        logging.info("Ti: {}, peak num: {}".format(t, len(peak_value)))
        return peak_value

    @staticmethod
    def _getDistri(nums):
        all_value = sorted(nums)
        return {
            "max": all_value[-1],
            "min": all_value[0],
            "avg": sum(all_value) / len(all_value),
            "mid": all_value[len(all_value) // 2],
            "p0.25": all_value[len(all_value) // 4],
            "p0.75": all_value[len(all_value) * 3 // 4],
        }

    def getModelDistri(self, t):
        return Env._getDistri(self.getAllValue(t))

    def getModelPeakDistri(self, t):
        return Env._getDistri(self.getAllPeakValue(t))


if (__name__ == "__main__"):
    import numpy as np
    pass
'''
    all_config.load()
    moniter.LogInit()
    logging.info("Start")
    N = 12
    k = 10
    T = 100
    env = Env(N, k, T)
    for i in range(1):
        feat = []
        value = []
        for j in range(2 ** N):
            feat.append(NKmodel.getGrayCode(N, j))
            value.append(env.getValue(x=j, t=i))
        #        moniter.DrawHist(point_pairs)
        moniter.Draw2DViaPCA(feat, value)
'''
