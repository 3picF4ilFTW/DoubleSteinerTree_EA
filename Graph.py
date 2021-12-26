import networkx as nx
import time
import numpy as np
import matplotlib.pyplot as plt
import utils
from copy import deepcopy


class Graph:
    def __init__(self):
        None

    def load_graph_from_file(self, filename):
        self.graph = nx.Graph()
        with open(filename) as f:  # opening the txt file
            lines = f.readlines()
        for i in range(len(lines)):  # deleting unnecessary information in the string and splitting into list
            lines[i] = lines[i].replace("\n", "").split(" ")
        self.abs_V = int(lines[0][1])
        self.abs_E = int(lines[0][2])
        self.abs_T_1 = int(lines[0][3])
        self.abs_T_2 = int(lines[0][4])
        self.alpha_1 = int(lines[1][1])
        self.alpha_2 = int(lines[1][2])
        self.gamma = int(lines[1][3])

        vertices = []
        for i in range(self.abs_V):
            vertices.append(i)

        edges = []
        self.terminal_1 = []
        self.terminal_2 = []
        for line in lines:
            if line[0] == "E":
                edges.append(utils.ordered_edge_weighted(int(line[1]), int(line[2]), int(line[3])))  # collects edges out of the file with weight
            if line[0] == "T1":
                self.terminal_1.append(int(line[1]))  # collects the Terminals1 out of file
            if line[0] == "T2":
                self.terminal_2.append(int(line[1]))

        self.graph.add_nodes_from(vertices)
        self.graph.add_weighted_edges_from(edges)

        for n1,n2,d in self.graph.edges.data():
            d["weight_1"] = self.alpha_1 * d["weight"]
            d["weight_2"] = self.alpha_2 * d["weight"]
