import random
from Graph import Graph
from AntColonyOptimization import AntColonyOptimization
from AntSolution import AntSolution
import time
import copy
import math
from multiprocessing import Process, Queue
import os
from datetime import datetime as dt


def create_graph(f : str):
    graph = Graph()
    graph.load_graph_from_file(f)
    return graph


def sample_params():
    ret = {}
    ret["tau_0"] = random.uniform(0.1 / (90000.0), 0.1 / (10000.0))
    ret["xi"] = random.uniform(0.05, 0.15)
    ret["q0"] = random.uniform(0.3, 0.9)
    ret["alpha"] = random.uniform(0.25, 2.0)
    ret["beta"] = random.uniform(0.25, 2.0)
    ret["rho"] = random.uniform(0.05, 0.5)
    ret["ants"] = random.randint(8, 32)
    return ret


def single_run(iteration, f : str, g : Graph, params, q : Queue):
    random.seed(1234)
    aco = AntColonyOptimization(g, params)
    sol, dur = aco.run(300, verbose=0)
    q.put({f : sol})
    
    f = "aco_tuning_out" + "/" + os.path.basename(f)
    output = f.replace(".txt", f"_{iteration}.txt")
    sol.write_solution_to_file(g, output)
    
    outputlog = f.replace(".txt", f"_{iteration}_log.txt")
    logfile = open(outputlog, "w")
    logfile.write(f"Log for instance {f}.\n\n")
    
    logfile.write(f"Used ({params}) and found best instance after {dur} seconds.\n")
    logfile.write(f"{sol}\n\n")


if __name__ == "__main__":
    random.seed(1234)
    #global graphs
    #global params
    
    files = list(map(lambda i : f"tuning_instances/inst{(i+1)*3:02}.txt", range(0, 10)))
    graphs = {f: create_graph(f) for f in files}
    
    best_values = {f: math.inf for f in files}
    best_params = {f: {} for f in files}
    
    total_best_value = math.inf
    total_best_param = {}
    
    q = Queue(10)
    param_runs = 85
    params_list = []

    for i in range(0, param_runs):
        params = sample_params()
        params_list.append(params)
    
        processes = []
        for f in files:
            p = Process(target=single_run, args=(i, f, graphs[f], params, q))
            processes.append(p)
            
        print(f"[{dt.now().strftime('%H:%M:%S')}] Starting run {i}...", flush=True)
        for p in processes:
            p.start()
        
        results = {}
        for j in range(0, 10):
            result = q.get(block=True)
            results = results | result
        
        print(f"[{dt.now().strftime('%H:%M:%S')}] Finished run {i}...", flush=True)
        
        for p in processes:
            p.join()
        
        total = 0
        for f in files:
            r = results[f]
            total += r.weight
            if r.weight < best_values[f]:
                best_values[f] = r.weight
                best_params[f] = i
            
        if total < total_best_value:
            total_best_value = total
            total_best_param = i
 
 
    file = open("aco_tuning.txt", "w")
    for p, i in zip(params_list, range(0, param_runs)):
        file.write(f"Param {i}: {p}\n")

    file.write("\n")
    for f in files:
        file.write(f"Best Parameters for {f} with {best_values[f]}: {best_params[f]}\n")

    file.write("\n")
    file.write(f"best overall Parameters with {total_best_value}: {total_best_param}\n")