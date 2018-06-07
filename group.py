import arg
import util.moniter
from util.config import all_config
import networkx as nx
from math import tanh


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
        self.relat.add_nodes_from(range(arg['AgentN']))
        self.power.add_nodes_from(range(arg['AgentN']))

    def random_init_relation(self):
        pass

    def random_init_power(self):
        pass

    def random_init(self):
        self.random_init_relation()
        self.random_init_power()

    def flat_init(self):
        for i in self.relat.node():
            for j in range(i):
                self.relat.add_weighted_edges_from([(i, j, 0.5)])
        for i in self.relat.node():
            for j in self.relat.node():
                self.power.add_weighted_edges_from([(i, j, 0.5)])

    def gen_relat_lk_graph(self, lk_func):
        self.relat_lk.clear()
        for u, v in self.relat.edges:
            if (lk_func(self.relat.edges[u][v])):
                self.relat_lk.add_edge(u, v)
        groups = [Group(g) for g in nx.connected_components(self.relat_lk)]
        return groups

    def relat_cd(self, cd_rate=0.95):
        for u, v in self.relat.edges:
            self.relat[u][v]['weight'] *= cd_rate
