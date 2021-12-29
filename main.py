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


def try_float(str):
    try:
        return float(str)
    except ValueError:
        return str

def main():
    if len(sys.argv) < 4:
        print("The following arguments are required: filename runtime algorithm parameters.")
        print("The algorithm-specific parameters can be provided as key=value pairs.")
        print("You may always use a paramter dir=... which sets an directory to output the solution to.")
        print(sys.argv[1])
        exit()

    else:
        input = sys.argv[1]
        limit = float(sys.argv[2])
        algorithm = sys.argv[3].lower()
        
        parameters = {key.lower(): try_float(value) for (key,value) in map(lambda s: s.split("="), sys.argv[4:])}    
        
    random.seed(1234)
    
    print(f"Loading input graph from {input}...")
    graph = Graph()
    graph.load_graph_from_file(input)
    
    start = time.process_time()
    if algorithm == "aco":
        print(f"Loading input graph from {input}...")
        graph = Graph()
        graph.load_graph_from_file(input)
        aco = AntColonyOptimization(graph, parameters)
        sol, dur = aco.run(limit, verbose=2)

    elif algorithm == "ea":
        print(f"Loading input graph from {input}...")
        graph = Graph2()
        graph.load_graph_from_file(input)
        ga = GA(graph)
        if parameters != {}:
            print(parameters)
            ga.setup(parameters)
        start_evol = time.process_time()
        best_val, best_sol, best_st = ga.do_evolution_single()
        end_evol = time.process_time()
        print(f"Best found value was {best_val}")
        print(f"Best found solutions were: \n{best_sol[0]}\n{best_sol[1]}")
        
        
        dur = None
        sol = AntSolution()
        sol.edges_1 = set(best_st[0].edges)
        sol.edges_2 = set(best_st[1].edges)
        sol.weight = best_val
        

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