from Graph import Graph
from AntSolution import AntSolution
import random
import utils
import time


class AntColonyOptimization:
    def __init__(self, g : Graph, params : dict[str, float]):
        self.graph = g
        
        # normalize discounts so we can use weight instead of weight_1 and weight_2
        # the reason for this is that if alpha_1 and alpha_2 differ by a lot this may skew the effect of pheromones
        self.discount_1 = float(self.graph.gamma) / float(self.graph.alpha_1)
        self.discount_2 = float(self.graph.gamma) / float(self.graph.alpha_2)
        
        self.tau0 = params.get("tau0", 1.0 / (30000.0 * 10.0))
        self.xi = params.get("xi", 0.1)
        self.q0 = params.get("q0", 0.75)
        self.alpha = params.get("alpha", 1.0)
        self.beta = params.get("beta", 1.0)
        self.rho = params.get("rho", 0.1)
        self.ants = int(params.get("ants", 16))

        # pheromone matrices based on ordered edges (tuples) 
        self.pheromones_1 = {}
        self.pheromones_2 = {}

        for v1,v2 in self.graph.graph.edges():
            edge = utils.ordered_edge(v1,v2)
            self.pheromones_1[edge] = self.tau0
            self.pheromones_2[edge] = self.tau0

    
    def run(self, max_duration, verbose=2):
        best_solution = None
        best_time = None
        
        start = time.process_time()
        while time.process_time() - start < max_duration:
            new_solutions, best_new_solution = self.run_ants()
            
            if verbose >= 2:
                print(f"New solution:\n{best_new_solution}")
            
            if best_solution is None or best_solution.weight > best_new_solution.weight:
                best_solution = best_new_solution
                best_time = time.process_time() - start
            
            if verbose >= 2:
                print(f"Best solution:\n{best_solution}")
                print(f"Duration: {time.process_time() - start} of {max_duration}\n")
        
            self.global_update(new_solutions, best_solution, verbose)
        
        return best_solution, best_time

    
    def run_ants(self):
        best_solution = None
        solutions = []
    
        for i in range(0, self.ants):
            ant = Ant(self)
            while not ant.done():
                ant.take_step()
            new_solution = ant.get_solution()
            solutions.append(new_solution)
            self.local_update(new_solution)
            
            if best_solution is None or best_solution.weight > new_solution.weight:
                best_solution = new_solution
        
        return solutions, best_solution


    def global_update(self, solutions, best_solution, verbose):
        sum1 = 0
        sum2 = 0
        for e in self.pheromones_1:
            sum1 += self.pheromones_1[e]
            self.pheromones_1[e] = (1.0 - self.rho) * self.pheromones_1[e]
        
        for e in self.pheromones_2:
            sum2 += self.pheromones_2[e]
            self.pheromones_2[e] = (1.0 - self.rho) * self.pheromones_2[e]
        
        for s in solutions:
            for e in s.edges_1:
                self.pheromones_1[e] += self.rho * pow(1.0 / (best_solution.weight), 2.0)
            for e in s.edges_2:
                self.pheromones_2[e] += self.rho * pow(1.0 / (best_solution.weight), 2.0)
        
        for e in best_solution.edges_1:
            self.pheromones_1[e] += self.rho * 1.0 / (best_solution.weight)
        
        for e in best_solution.edges_2:
            self.pheromones_2[e] += self.rho * 1.0 / (best_solution.weight)
        
        if verbose >= 2:
            print(f"SUMS: {sum1} {sum2}")
    
    
    def local_update(self, solution):
        for e in solution.edges_1:
            self.pheromones_1[e] = (1.0 - self.xi) * self.pheromones_1[e] + self.xi * self.tau0
        
        for e in solution.edges_2:
            self.pheromones_2[e] = (1.0 - self.xi) * self.pheromones_2[e] + self.xi * self.tau0
    

class Ant:
    def __init__(self, aco : AntColonyOptimization):
        self.aco = aco
        
        # terminals yet to be traversed
        self.terminals_1 = set(self.aco.graph.terminal_1)
        self.terminals_2 = set(self.aco.graph.terminal_2)

        # start terminals
        n1 = random.choice(self.aco.graph.terminal_1)
        n2 = random.choice(self.aco.graph.terminal_2)
        self.terminals_1.remove(n1)
        self.terminals_2.remove(n2)

        # nodes already traversed (+ order in terms of neighbors for trimming)
        self.nodes_1 = {n1 : 0}
        self.nodes_2 = {n2 : 0}

        # edges already traversed
        self.edges_1 = set()
        self.edges_2 = set()
        self.edges_shared = set()
        
        # keeping track of weights
        self.weight_1 = 0
        self.weight_2 = 0
        self.weight_shared = 0
        
        # candidate edges
        self.cand_1 = {utils.ordered_edge(n1, n) for n in self.aco.graph.graph.neighbors(n1)}
        self.cand_2 = {utils.ordered_edge(n2, n) for n in self.aco.graph.graph.neighbors(n2)}
    
    
    def done(self):
        return len(self.terminals_1) + len(self.terminals_2) == 0
    
    
    def take_step(self):
        if len(self.terminals_1) > 0:        
            # pick an edge
            ranking = list(map(lambda e: (e, self.compute_edge_objective(e, self.aco.pheromones_1, self.edges_2, self.aco.discount_1)), self.cand_1))
            q = random.random()
            if q < self.aco.q0:
                edge = max(ranking, key=lambda t: t[1])[0]
            else:
                ranking = list(zip(*ranking)) # unzips the ranking list
                edge = random.choices(ranking[0], weights=ranking[1])[0]
                
            # add edge
            self.edges_1.add(edge)
            self.weight_1 += self.aco.graph.graph.get_edge_data(*edge)["weight_1"]
            
            if edge in self.edges_2 and edge not in self.edges_shared:
                self.edges_shared.add(edge)
                self.weight_shared += self.aco.graph.graph.get_edge_data(*edge)["weight"]
            
            # update data structures
            new_node = edge[0] if edge[0] not in self.nodes_1 else edge[1]
            self.nodes_1[edge[0]] = self.nodes_1.get(edge[0], 0) + 1
            self.nodes_1[edge[1]] = self.nodes_1.get(edge[1], 0) + 1
            
            self.terminals_1.discard(new_node)
            
            cand_add = {utils.ordered_edge(new_node, n) for n in self.aco.graph.graph.neighbors(new_node)}
            self.cand_1.update(cand_add)
            cand_del = {e for e in self.cand_1 if e[0] in self.nodes_1 and e[1] in self.nodes_1}
            self.cand_1.difference_update(cand_del)

        if len(self.terminals_2) > 0:        
            # pick an edge
            ranking = list(map(lambda e: (e, self.compute_edge_objective(e, self.aco.pheromones_2, self.edges_1, self.aco.discount_2)), self.cand_2))
            q = random.random()
            if q < self.aco.q0:
                edge = max(ranking, key=lambda t: t[1])[0]
            else:
                ranking = list(zip(*ranking)) # unzips the ranking list
                edge = random.choices(ranking[0], weights=ranking[1])[0]
                
            # add edge
            self.edges_2.add(edge)
            self.weight_2 += self.aco.graph.graph.get_edge_data(*edge)["weight_2"]
            
            if edge in self.edges_1 and edge not in self.edges_shared:
                self.edges_shared.add(edge)
                self.weight_shared += self.aco.graph.graph.get_edge_data(*edge)["weight"]
            
            # update data structures
            new_node = edge[0] if edge[0] not in self.nodes_2 else edge[1]
            self.nodes_2[edge[0]] = self.nodes_2.get(edge[0], 0) + 1
            self.nodes_2[edge[1]] = self.nodes_2.get(edge[1], 0) + 1
            
            self.terminals_2.discard(new_node)
            
            cand_add = {utils.ordered_edge(new_node, n) for n in self.aco.graph.graph.neighbors(new_node)}
            self.cand_2.update(cand_add)
            cand_del = {e for e in self.cand_2 if e[0] in self.nodes_2 and e[1] in self.nodes_2}
            self.cand_2.difference_update(cand_del)
        
    
    def compute_edge_objective(self, edge, pheromones, discounted_edges, discount):
        f1 = pheromones[edge] ** self.aco.alpha
        weight = self.aco.graph.graph.get_edge_data(*edge)["weight"]
        if edge in discounted_edges:
            weight += discount
            if weight == 0:
                f2 = 1.0
            else:
                f2 = ((1.0 / weight) ** self.aco.beta)
        else:
            f2 = ((1.0 / weight) ** self.aco.beta)
        return f1 * f2
    
    
    def trim(self):
        changed = True
        while changed:
            changed = False
            edges_to_remove = set()
            
            for n,v in self.nodes_1.items():
                if n not in self.aco.graph.terminal_1 and v == 1:
                    edges_to_remove.update({e for e in self.edges_1 if e[0] == n or e[1] == n})
                    changed = True
            
            for e in edges_to_remove:
                self.weight_1 -= self.aco.graph.graph.get_edge_data(*e)["weight_1"]
                self.nodes_1[e[0]] -= 1
                self.nodes_1[e[1]] -= 1
                if e in self.edges_shared:
                    self.weight_shared -= self.aco.graph.graph.get_edge_data(*e)["weight"]
            
            self.edges_1.difference_update(edges_to_remove)
            self.edges_shared.difference_update(edges_to_remove)
            
        changed = True
        while changed:
            changed = False
            edges_to_remove = set()
            
            for n,v in self.nodes_2.items():
                if n not in self.aco.graph.terminal_2 and v == 1:
                    edges_to_remove.update({e for e in self.edges_2 if e[0] == n or e[1] == n})
                    changed = True
            
            for e in edges_to_remove:
                self.weight_2 -= self.aco.graph.graph.get_edge_data(*e)["weight_2"]
                self.nodes_2[e[0]] -= 1
                self.nodes_2[e[1]] -= 1
                if e in self.edges_shared:
                    self.weight_shared -= self.aco.graph.graph.get_edge_data(*e)["weight"]
            
            self.edges_2.difference_update(edges_to_remove)
            self.edges_shared.difference_update(edges_to_remove)
    
    
    def get_solution(self):
        self.trim()
        
        solution = AntSolution()
        solution.edges_1 = self.edges_1
        solution.edges_2 = self.edges_2
        solution.weight = self.weight_1 + self.weight_2 + self.weight_shared * self.aco.graph.gamma
        
        return solution
        