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


def main():
    if len(sys.argv) < 4:
        print("The following arguments are required: filename runtime algorithm parameters.")
        print("The algorithm-specific parameters can be provided as key=value pairs.")
        exit()

    else:
        input = sys.argv[1]
        limit = float(sys.argv[2])
        algorithm = sys.argv[3].lower()
        
        parameters = {key: float(value) for (key,value) in map(lambda s: s.split("="), sys.argv[4:])}    
        
    random.seed(1234)
    
    print(f"Loading input graph from {input}...")
    graph = Graph()
    graph.load_graph_from_file(input)
    
    start = time.process_time()
    if algorithm == "aco":
        # sol, dur = ...
        pass
    elif algorithm == "ea":
        # sol, dur = ...
        pass
    else:
        print(f"Invalid algorithm {algorithm}!")
    end = time.process_time()

    print(f"Running time was: {end - start}")
    
    output = input.replace(".txt", f"_{algorithm}.txt")
    outputlog = input.replace(".txt", "_log.txt")

    sol.write_solution_to_file(graph, output)
    
    f = open(outputlog, "w")
    f.write(f"Log for instance {input}.\n\n")
    
    f.write(f"{algorithm} used ({parameters}) and took {end-start} seconds. The best instance was found after {dur} seconds.\n")
    f.write(f"{sol}\n\n")


if __name__ == "__main__":
    main()
    #runall_with_grid()