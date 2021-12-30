import sys
import time
import itertools
import random
import math
import bisect
from copy import copy
import networkx as nx
import utils
from Graph import Graph
from AntSolution import AntSolution
from AntColonyOptimization import AntColonyOptimization
from GA2 import GA
from Graph_GA import Graph as Graph2
from main import main
import os

#os.system("main.py .\\instances\\0011.txt 1 ea max_time=1")
for i in range(30):
    inst = i+1
    if inst < 10:
        instance = f".\\tuning_instances\\inst0{inst}.txt"
    else:
        instance = f".\\tuning_instances\\inst{inst}.txt"

    os.system(f"main.py {instance} 300 ea max_time=300 population_size=20 parental_survivors=10 mutation_chance=0.001 tournament_size=5 max_generations_with_no_improvement=5")

