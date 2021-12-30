import random
from GA2 import GA
from Graph_GA import Graph as Graph2
import time
from AntSolution import AntSolution
import copy
import numpy as np
import logging


def time_till_best(eval_lst_in_each_generation, time_lst_in_each_generation):
    if not len(eval_lst_in_each_generation) - len(time_lst_in_each_generation) == 1:
        print("something wrong with inputs not according to size")
        exit(-1)
    minimum = min(eval_lst_in_each_generation)
    for index, val in enumerate(eval_lst_in_each_generation):
        if val == minimum:
            if index == 0:
                return 0, 0
            else:
                return time_lst_in_each_generation[index - 1], index
def monte_carlo_tuning(instance, num_paras_to_eval, time_constant, min_gen_per_parameter):
    print("Initialization:")
    possible_parameters = {
        "mutation_chance" : [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
        "crossover_choice" : ["uniform", "onepoint"],
        "max_generations_with_no_improvement" : [3,4,5],
        "tournament_size" : [1,2,3,10,20] #k will always be 1
    }

    different_solutions_to_evaluate = num_paras_to_eval
    time_constant = time_constant
    gen_per_parameter = min_gen_per_parameter



    input = instance
    print(f"Loading input graph from {input}...")
    graph = Graph2()
    graph.load_graph_from_file(input)
    ga = GA(graph)

    parameters = {
        "max_time" : 0,
        "population_size" : 1,
        "parental_survivors" : 1
    }
    ga.setup(parameters)

    start_evol = time.process_time()
    best_val, best_sol, best_st, eval_lst_in_each_generation, time_lst_in_each_generation = ga.do_evolution_single_for_tuning(verbose=0)
    end_evol = time.process_time()
    print(f"Too evaluate a single solution takes {time_lst_in_each_generation[0]} which means in {time_constant/60} minutes we will be able to solve about {time_constant/time_lst_in_each_generation[0]} instances")
    print(f"We try to evaluate {different_solutions_to_evaluate} different parameters which means each parameter can approximately solve {time_constant/time_lst_in_each_generation[0]/different_solutions_to_evaluate/2} by a conservative estimate")
    max_population = time_constant/time_lst_in_each_generation[0]/different_solutions_to_evaluate/gen_per_parameter/2
    population_size = int(min(100, max_population))
    parental_survivors = int(population_size/2)
    print(f"We set the populations_size so that at least {gen_per_parameter} generations per parameter can be solved but not more than 100 parents as {population_size}")

    parameters = {
        "max_time" : time_constant/different_solutions_to_evaluate,
        "population_size" : population_size,
        "parental_survivors" : parental_survivors
    }
    best_overall_dict = {
        "best_val" : np.inf,
        "time_till_best": np.inf,
        "best_sol" : None,
        "best_st" : None,
        "best_parameters" : None
    }

    random.seed(123)
    start = time.process_time()
    for i in range(different_solutions_to_evaluate): #this uniformly chooses from the possible parameter list so this is montecarlo sampling
        for key in possible_parameters:
            choosen = random.choice(possible_parameters[key])
            if key == "tournament_size":
                itr_ctr = 0
                while choosen > population_size:
                    itr_ctr += 1
                    choosen = random.choice(possible_parameters[key]) #for tournament size only choose up to population size
                    if itr_ctr > 10000:
                        print(f"not possible to choose torunament size from {possible_parameters[key]} \nwith a pop size of {population_size}")
                        exit(-1)
            parameters[key] = choosen

        ga.setup(parameters)
        best_val, best_sol, best_st, eval_lst_in_each_generation, time_lst_in_each_generation = ga.do_evolution_single_for_tuning(verbose=0)
        time_until_best_val_was_found, gens = time_till_best(eval_lst_in_each_generation, time_lst_in_each_generation)

        if best_overall_dict["best_val"] >= best_val :
            best_overall_dict["best_val"] = best_val
            best_overall_dict["generations_till_best"] = gens
            best_overall_dict["time_till_best"] = time_until_best_val_was_found
            best_overall_dict["best_sol"] = best_sol
            best_overall_dict["best_st"] = best_st
            best_overall_dict["best_parameters"] = copy.deepcopy(parameters)

        if best_overall_dict["best_val"] == best_val and best_overall_dict["time_till_best"] > time_until_best_val_was_found:
            best_overall_dict["best_val"] = best_val
            best_overall_dict["generations_till_best"] = gens
            best_overall_dict["time_till_best"] = time_until_best_val_was_found
            best_overall_dict["best_sol"] = best_sol
            best_overall_dict["best_st"] = best_st
            best_overall_dict["best_parameters"] = copy.deepcopy(parameters)
        break


    end = time.process_time()
    best_val = best_overall_dict["best_val"]
    best_sol = best_overall_dict["best_sol"]
    time_till = best_overall_dict["time_till_best"]
    gens = best_overall_dict["generations_till_best"]
    parameters = best_overall_dict["best_parameters"]

    logging.basicConfig(level=logging.DEBUG, filename="logfile.txt", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    logging.info(f"\n\n\nSolution for instance {instance}:")
    logging.info(f"The tuning process took {end-start} seconds")
    logging.info(f"Best found value was {best_val}")
    logging.info(f"The value was found after {time_till} seconds and {gens} generations")
    logging.info(f"The parameters that were used are \n{parameters}")
    logging.info(f"Best found solutions were: \n{best_sol[0]}\n{best_sol[1]}")


    AntSol = AntSolution()
    AntSol.edges_1 = set(best_overall_dict["best_st"][0].edges)
    AntSol.edges_2 = set(best_overall_dict["best_st"][1].edges)

    output = instance.split(".txt")[0]+"_ea.txt"
    AntSol.write_solution_to_file(graph,output)


if __name__ == "__main__":


    for i in range(10):
        inst = int((i+1)*3)
        if inst < 10:
            instance = f".\\tuning_instances\\inst0{inst}.txt"
        else:
            instance = f".\\tuning_instances\\inst{inst}.txt"

        start = time.process_time()
        monte_carlo_tuning(instance, 10, 40, 5)
        end = time.process_time()
        print(f"monte carlo tuning took {end-start}")
        break

