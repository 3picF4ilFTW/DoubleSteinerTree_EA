from Graph_GA import Graph
import random
import copy
import timeit
import networkx as nx
import numpy as np
import utils


class GA:
    def __init__(self, graph:Graph):
        self.population_size = 100
        self.mutation_chance = 0.01

        self.parental_survivors = 50 #for tournament and roulettewheel selection
        self.tournament_size = 2

        self.parental_choice = "tournament" # or roulette
        self.crossover_choice = "uniform" # or onepoint

        self.max_generations_with_no_improvement = 3
        self.max_time = 3
        self.already_solved_total = {} #in this dictionary save all already solved instances
        #saves in tuple (st1, st2, eval)

        self.graph = graph
        self.solution1 = []
        self.solution2 = []
        for i in range(self.population_size):
            self.solution1.append([0] * len(graph.graph))# initiallize the solutions empty
            self.solution2.append([0] * len(graph.graph))# this means empty lists of len size graph

        self.current_solution1 = [0] * len(graph.graph)
        self.current_solution2 = [0] * len(graph.graph)
        self.current_st1 = None
        self.current_st2 = None



    """
    takes the already initialized solutions of the init and returns a random 
    solution such that the chance of value becoming 1 is the chance value 
    given to the function.
    This function does this for all solutions
    """
    #already tested works correctly
    def random_solution_all(self, chance = 0.3):
        for index, solution in enumerate(self.solution1):
            if index == 0:
                continue #do this to keep 1 solution 0 to see if the greedy solution performs best
            for index2, val2 in enumerate(solution):
                if random.random() < chance:
                    self.solution1[index][index2] = 1
        for index, solution in enumerate(self.solution2):
            if index == 0:
                continue #do this to keep 1 solution 0 to see if the greedy solution performs best
            for index2, val2 in enumerate(solution):
                if random.random() < chance:
                    self.solution2[index][index2] = 1

    """
        If just a single solution shall be randomized use the following function
    """
    #tested works correctly
    def random_solution_single(self, chance = 0.3):
        sol = [0] * len(self.graph.graph)
        for index , val in enumerate(sol):
            if random.random() < chance:
                sol[index] = 1
        return sol
    """
        Alters the value in a given solution with chance
    """
    #tested works correclty
    def mutation(self, solution):
        for index, val in enumerate(solution):
            if random.random() < self.mutation_chance:
                solution[index] = 1 - solution[index] #gives 1 if the solution was 0 and 0 if the solution was 1
        return solution
    """
        does the crossover operation specified in crossover_type for given solution1 and solution2
    """
    #one point works correctly
    def crossover(self, solution1, solution2):
        if self.crossover_choice == "onepoint":
            crossover_point = random.randint(1,len(solution2)-1)
            solution = solution1[0:crossover_point]
            solution = solution + solution2[crossover_point: len(solution2)]
            return solution
        elif self.crossover_choice == "twopoint":
            print("not yet implemented")
        elif self.crossover_choice == "uniform":
            new_sol = []
            for index in range(len(solution1)):
                if random.random() < 0.5:
                    new_sol.append(solution1[index])
                else:
                    new_sol.append(solution2[index])
            
            return new_sol
        else:
            print(f"not specified operation {self.crossover_choice}")
            return None

    """
        Returns a list of nodes built from the binary solution vector
    """
    def binary_solution_to_node_list(self, solution):
        node_list = []
        for index, val  in enumerate(solution):
            if val == 1:
                node_list.append(copy.copy(index))
        return node_list
    """
        Returns the steiner tree built from the solution vector
    """



    """
        Returns the weight of a solution in steiner_tree1 and steiner_tree2 and overlap.
        overlap gives the penalty or 
    """
    def total_weight_in_a_graph(self, graph: nx.Graph):
        w = 0
        
    
    def evaluate(self, steiner_tree1: nx.Graph, steiner_tree2: nx.Graph):
        new_s1_size = 0
        att = nx.get_edge_attributes(self.graph.graph, "weight")
        
        for edge in steiner_tree1.edges:
            new_s1_size += att[utils.ordered_edge(edge[0],edge[1])]
        new_s1_size = new_s1_size*self.graph.alpha_1
        
        new_s2_size = 0
        for edge in steiner_tree2.edges:
            new_s2_size += att[utils.ordered_edge(edge[0],edge[1])]
        new_s2_size = new_s2_size*self.graph.alpha_2
        overlap = 0
        shared_edges = set(steiner_tree1.edges).intersection(set(steiner_tree2.edges))
        for edge in shared_edges:
            overlap += att[utils.ordered_edge(edge[0],edge[1])] * self.graph.gamma
        total = new_s1_size + new_s2_size + overlap
        
        
        return total
        
    

    def find_steiner_tree(self, graph: Graph, terminal_nodes, treeindex: int, other_st: nx.Graph, dg: nx.Graph):
        solutions = getattr(self, f"solution{treeindex}")

        best_eval = np.inf
        best_st = None
        best_sol = None
        eval_lst = []
        already_solved_dict = {}
        for sol in solutions:
            """#I think this might cause problems omit for now
            if treeindex == 2:
                if (tuple(self.current_solution1), tuple(sol)) in self.already_solved_total:
                    eval_lst.append(self.already_solved_total[(tuple(self.current_solution1), tuple(sol))])
                    continue
            else:
                if (tuple(sol), tuple(self.current_solution2)) in self.already_solved_total:
                    eval_lst.append(self.already_solved_total[(tuple(sol), tuple(self.current_solution2))])
                    continue
            """
            node_lst = self.binary_solution_to_node_list(sol)
            tmp_dg = copy.deepcopy(dg)
            for node in node_lst:
                if not node in tmp_dg:
                    graph.augment_distance_graph(tmp_dg, node, f"weight_{treeindex}_mod", already_solved_dict= already_solved_dict)

            st_tmp = graph.rebuild_steiner_tree(tmp_dg, treeindex)
            """#This would be nice if it worked such that we can remove the deepcopy but first we would have to check whether a key node is a terminal node
            for node in node_lst:
                if node in tmp_dg:
                    tmp_dg.remove_node(node)"""
            if other_st is None:
                return None,sol,  st_tmp, eval_lst
            eval = self.evaluate(st_tmp, other_st)
            eval_lst.append(eval)
            """#I think this might cause problems omit for now
            if treeindex == 2:
                self.already_solved_total[(tuple(self.current_solution1), tuple(sol))] = eval
            else:
                self.already_solved_total[(tuple(sol), tuple(self.current_solution2))] = eval
            """
            if eval < best_eval:
                best_eval = eval
                best_st = st_tmp
                best_sol = sol
        return best_eval,best_sol, best_st, eval_lst



    """
        Just initializes the solutions and builds the steiner trees from them
    """
    def initialize_solution(self):
        self.random_solution_all(self.mutation_chance)
        dg = self.graph.distance_graph(self.graph.terminal_1, 1, True)
        tmp,sol1, st1, lst = self.find_steiner_tree(self.graph, self.graph.terminal_1,1, None, dg)
        self.graph.modify_weights(st1, 1)
        dg = self.graph.distance_graph(self.graph.terminal_2, 2, True)
        eval,sol2, st2, eval_lst2 = self.find_steiner_tree(self.graph, self.graph.terminal_2,2,st1, dg)
        self.graph.modify_weights(st2, 2)
        dg = self.graph.distance_graph(self.graph.terminal_1, 1, True)
        eval, sol1, st1, eval_lst1 = self.find_steiner_tree(self.graph, self.graph.terminal_1, 1, other_st=st2, dg=dg)


        return eval, (sol1, sol2), (st1, st2), eval_lst1, eval_lst2

    def select_parents(self, solution_index: int, eval_lst: list):
        if self.parental_choice == "tournament":
            solution = getattr(self, f"solution{solution_index}")
            parent_index_list = []
            possible_parents = list(range(self.population_size))
            while len(parent_index_list) < self.parental_survivors: #do selection until enough parents are selected
                tournament_participants = []
                for i in range(self.tournament_size):
                    tournament_participants.append(random.choice(possible_parents))
                best_t_val = np.inf
                tournament_winner = 0
                for t in tournament_participants:
                    t_val = eval_lst[t]
                    if t_val < best_t_val:
                        best_t_val = t_val
                        tournament_winner = t

                parent_index_list.append(tournament_winner)
            self.parents = []
            for parent in parent_index_list:
                self.parents.append(solution[parent])
            return self.parents

        if self.parental_choice == "roulette":
            print("not yet implemented")

    def create_children(self, parents):
        children = []
        while len(children) < self.population_size-self.parental_survivors:
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            children.append(self.crossover(p1, p2))

        self.parents = parents + children #append the children to the parent list
        for index, parent in enumerate(self.parents): #mutate everything in the parentlist
            self.parents[index] = self.mutation(parent)

    def do_evolution_single(self, verbose=2):
        random.seed(123)
        best_so_far, best_sol_tuple, best_st_tuple, eval_list1, eval_list2 = self.initialize_solution()
        self.current_solution1, self.current_solution2 = copy.deepcopy(best_sol_tuple[0]), copy.deepcopy(
            best_sol_tuple[1])
        self.current_st1, self.current_st2 = copy.deepcopy(best_st_tuple[0]), copy.deepcopy(best_st_tuple[1])

        start = timeit.default_timer()
        start_last_improvement = timeit.default_timer()
        gen_ctr = 0
        total_gen = 0
        while True:  # this loop iteratively evolves sol1 or sol2 until its max is reached
            end_total_timer = timeit.default_timer()
            """if end_total_timer - start_last_improvement > 60:
                print("no improvement for 60 seconds break")
                break"""
            while True:
                gen_ctr += 1
                total_gen += 1
                parents = self.select_parents(2, eval_list2)
                self.create_children(parents)
                self.solution2 = self.parents
                self.graph.reset_edge_weights(2)
                self.graph.modify_weights(self.current_st1, 1)
                dg = self.graph.distance_graph(self.graph.terminal_2, 2, True)
                tmp_best_eval, tmp_best_sol, tmp_best_st, tmp_eval_lst = self.find_steiner_tree(self.graph,
                                                                                                self.graph.terminal_2,
                                                                                                2, self.current_st1, dg)
                eval_list2 = tmp_eval_lst
                if tmp_best_eval < best_so_far:
                    start_last_improvement = timeit.default_timer()
                    best_so_far = tmp_best_eval
                    self.current_solution2 = copy.deepcopy(tmp_best_sol)
                    self.current_st2 = copy.deepcopy(tmp_best_st)
                    gen_ctr = 0
                    break
                if gen_ctr > self.max_generations_with_no_improvement:
                    gen_ctr = 0
                    if tmp_best_eval <= best_so_far:
                        start_last_improvement = timeit.default_timer()
                        best_so_far = tmp_best_eval
                        self.current_solution2 = copy.deepcopy(tmp_best_sol)
                        self.current_st2 = copy.deepcopy(tmp_best_st)
                        gen_ctr = 0
                        break
                    break
            end = timeit.default_timer()
            if verbose > 0:
                print(
                    f"After {total_gen} generations and a time of {end - start} the solutions are:\n{self.current_solution1}\n{self.current_solution2}")
                print(f"Best eval is: {best_so_far}")
            if end - start > self.max_time:
                break
            gen_ctr = 0
            while True:
                gen_ctr += 1
                total_gen += 1
                parents = self.select_parents(1, eval_list1)
                self.create_children(parents)
                self.solution1 = self.parents
                self.graph.reset_edge_weights(1)
                self.graph.modify_weights(self.current_st2, 2)
                dg = self.graph.distance_graph(self.graph.terminal_1, 1, True)
                tmp_best_eval, tmp_best_sol, tmp_best_st, tmp_eval_lst = self.find_steiner_tree(self.graph,
                                                                                                self.graph.terminal_1,
                                                                                                1, self.current_st2, dg)
                eval_list1 = tmp_eval_lst
                if tmp_best_eval < best_so_far:
                    start_last_improvement = timeit.default_timer()
                    best_so_far = tmp_best_eval
                    self.current_solution1 = copy.deepcopy(tmp_best_sol)
                    self.current_st1 = copy.deepcopy(tmp_best_st)
                    gen_ctr = 0
                    break
                if gen_ctr > self.max_generations_with_no_improvement:
                    gen_ctr = 0
                    if tmp_best_eval <= best_so_far:
                        start_last_improvement = timeit.default_timer()
                        best_so_far = tmp_best_eval
                        self.current_solution1 = copy.deepcopy(tmp_best_sol)
                        self.current_st1 = copy.deepcopy(tmp_best_st)
                        gen_ctr = 0
                        break
                    break

            end = timeit.default_timer()
            if verbose > 0:
                print(
                    f"After {total_gen} generations and a time of {end - start} the solutions are:\n{self.current_solution1}\n{self.current_solution2}")
                print(f"Best eval is: {best_so_far}")
            if end - start > self.max_time:
                break

        return best_so_far, best_sol_tuple, best_st_tuple


    def do_evolution_single_for_tuning(self, verbose = 2):
        random.seed(123)
        best_so_far, best_sol_tuple, best_st_tuple, eval_list1, eval_list2 = self.initialize_solution()
        self.current_solution1, self.current_solution2 = copy.deepcopy(best_sol_tuple[0]), copy.deepcopy(best_sol_tuple[1])
        self.current_st1, self.current_st2 = copy.deepcopy(best_st_tuple[0]), copy.deepcopy(best_st_tuple[1])

        start = timeit.default_timer()
        start_last_improvement = timeit.default_timer()

        eval_in_each_generation = [best_so_far] #this list will be longer than the time list as it contains the initial value
        time_taken_for_each_generation = []

        gen_ctr = 0
        total_gen = 0
        while True:#this loop iteratively evolves sol1 or sol2 until its max is reached
            end_total_timer = timeit.default_timer()
            """if end_total_timer - start_last_improvement > 60:
                print("no improvement for 60 seconds break")
                break"""
            while True:
                gen_ctr += 1
                total_gen += 1
                parents = self.select_parents(2, eval_list2)
                self.create_children(parents)
                self.solution2 = self.parents
                self.graph.reset_edge_weights(2)
                self.graph.modify_weights(self.current_st1, 1)
                dg = self.graph.distance_graph(self.graph.terminal_2, 2, True)
                tmp_best_eval,tmp_best_sol, tmp_best_st, tmp_eval_lst = self.find_steiner_tree(self.graph, self.graph.terminal_2,2, self.current_st1, dg)
                eval_list2 = tmp_eval_lst

                end_total_timer = timeit.default_timer() #appends the eval of a generation and its time
                eval_in_each_generation.append(tmp_best_eval)
                time_taken_for_each_generation.append(end_total_timer - start)

                if tmp_best_eval <= best_so_far:
                    start_last_improvement = timeit.default_timer()
                    best_so_far = tmp_best_eval
                    self.current_solution2 = copy.deepcopy(tmp_best_sol)
                    self.current_st2 = copy.deepcopy(tmp_best_st)
                    gen_ctr = 0
                    break
                if gen_ctr > self.max_generations_with_no_improvement:
                    gen_ctr = 0
                    break
                end = timeit.default_timer()
                if end - start > self.max_time:
                    break
            end = timeit.default_timer()
            if verbose > 0:
                print(f"After {total_gen} generations and a time of {end-start} the solutions are:\n{self.current_solution1}\n{self.current_solution2}")
                print(f"Best eval is: {best_so_far}")
            if end-start > self.max_time:
                break
            gen_ctr = 0
            while True:
                gen_ctr += 1
                total_gen += 1
                parents = self.select_parents(1, eval_list1)
                self.create_children(parents)
                self.solution1 = self.parents
                self.graph.reset_edge_weights(1)
                self.graph.modify_weights(self.current_st2, 2)
                dg = self.graph.distance_graph(self.graph.terminal_1, 1, True)
                tmp_best_eval,tmp_best_sol, tmp_best_st, tmp_eval_lst = self.find_steiner_tree(self.graph, self.graph.terminal_1,1, self.current_st2, dg)
                eval_list1 = tmp_eval_lst

                end_total_timer = timeit.default_timer()  # appends the eval of a generation and its time
                eval_in_each_generation.append(tmp_best_eval)
                time_taken_for_each_generation.append(end_total_timer - start)

                if tmp_best_eval <= best_so_far:
                    start_last_improvement = timeit.default_timer()
                    best_so_far = tmp_best_eval
                    self.current_solution1 = copy.deepcopy(tmp_best_sol)
                    self.current_st1 = copy.deepcopy(tmp_best_st)
                    gen_ctr = 0
                    break
                if gen_ctr > self.max_generations_with_no_improvement:
                    gen_ctr = 0
                    break
                end = timeit.default_timer()
                if end - start > self.max_time:
                    break

            end = timeit.default_timer()
            if verbose > 0:
                print(
                    f"After {total_gen} generations and a time of {end - start} the solutions are:\n{self.current_solution1}\n{self.current_solution2}")
                print(f"Best eval is: {best_so_far}")
            if end-start > self.max_time:
                break


        return best_so_far, best_sol_tuple, best_st_tuple, eval_in_each_generation, time_taken_for_each_generation



    """
        self.population_size = 20
        self.mutation_chance = 0.1

        self.parental_survivors = 10 #for tournament and roulettewheel selection
        self.tournament_size = 2

        self.parental_choice = "tournament" # or roulette
        self.crossover_choice = "onepoint" # or two point or circle crossover not yet implemented

        self.max_generations_with_no_improvement = 3
        self.max_time = 3

    """
    def setup(self, parameters: dict):
        for varname in parameters:
            if varname == "population_size" or varname == "parental_survivors" or varname == "tournament_size"  or varname == "max_generations_with_no_improvement":
                setattr(self, f"{varname}", int(parameters[varname]))
            elif varname == "parental_choice" or varname == "crossover_choice":
                setattr(self, f"{varname}", str(parameters[varname]))
                if varname == "parental_choice":
                    if getattr(self, f"{varname}") != "tournament" or getattr(self, f"{varname}") != "roulette":
                        print("Error in setup as expected")
                        exit(-1)

            else:
                setattr(self, f"{varname}", parameters[varname])







        self.solution1 = []
        self.solution2 = []
        for i in range(self.population_size):
            self.solution1.append([0] * len(self.graph.graph))  # initiallize the solutions empty
            self.solution2.append([0] * len(self.graph.graph))  # this means empty lists of len size graph

        self.current_solution1 = [0] * len(self.graph.graph)
        self.current_solution2 = [0] * len(self.graph.graph)
        self.current_st1 = None
        self.current_st2 = None
