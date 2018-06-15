# -*- coding:utf-8 -*-
import arg
import util.moniter
from util.config import all_config
import networkx as nx
from math import tanh
from random import uniform as unif


class Group:
    def __init__(self, node_set={}):
        self.nodes = node_set

    def has(self, x):
        return x in self.nodes

    def __len__(self):
        return len(self.nodes)


class SoclNet:
    def __init__(self, arg):
        self.arg = arg
        self.relat = nx.Graph()
        self.relat_lk = nx.Graph()
        self.power = nx.DiGraph()
        self.relat.add_nodes_from(range(arg['Nagent']))
        self.power.add_nodes_from(range(arg['Nagent']))

    def random_init_relation(self):
        pass

    def random_init_power(self):
        pass

    def random_init(self):
        self.random_init_relation()
        self.random_init_power()

    # 初始关系数值为0.5加随机扰动
    def flat_init(self):
        # 无向Graph自动补全
        for i in self.relat.node():
            for j in range(i):
                self.relat.add_weighted_edges_from([(i, j, 0.5 + unif(-0.01, 0.01))])
        for i in self.power.node():
            for j in self.power.node():
                self.power.add_weighted_edges_from([(i, j, 0.5 + unif(-0.01, 0.01))])

    def custom_init(self, relat_m, power_m):
        for i in range(self.arg['Nagent']):
            for j in range(self.arg['Nagent']):
                self.relat.add_weighted_edges_from([(i, j, relat_m[i][j])])
        for i in range(self.arg['Nagent']):
            for j in range(self.arg['Nagent']):
                self.power.add_weighted_edges_from([(i, j, power_m[i][j])])

    def gen_relat_lk_graph(self, lk_func):
        self.relat_lk.clear()
        for u, v in self.relat.edges:
            if lk_func(self.relat.edges[u][v]):
                self.relat_lk.add_edge(u, v)
        groups = [Group(g) for g in nx.connected_components(self.relat_lk)]
        return groups

    def relat_cd(self, cd_rate=0.95):
        for u, v in self.relat.edges:
            self.relat[u][v]['weight'] *= cd_rate

    def power_delta(self, u, v, delta):
        self.power[u][v]['weight'] += delta
        if self.power[u][v]['weight'] < 0:
            self.power[u][v]['weight'] = 0
        if self.power[u][v]['weight'] > 1:
            self.power[u][v]['weight'] = 1

    def power_relat(self, u, v, delta):
        self.relat[u][v]['weight'] += delta
        if self.relat[u][v]['weight'] < 0:
            self.relat[u][v]['weight'] = 0
        if self.relat[u][v]['weight'] > 1:
            self.relat[u][v]['weight'] = 1

    def get_power_close_centrality(self):
        for u in self.power.nodes:
            for v in self.power.nodes:
                self.power[u][v]['dist'] = self.arg['pow_w2d'](self.power[u][v]['weight'])
        return nx.closeness_centrality(G=self.power, distance='dist')
