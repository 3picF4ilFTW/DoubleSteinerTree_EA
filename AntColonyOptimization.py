from Graph import Graph
import random
import utils


class AntColonyOptimization:
    def __init__(self, g : Graph, params : dict[str, float]):
        self.graph = g
        
        # normalize discounts so we can use weight instead of weight_1 and weight_2
        # the reason for this is that if alpha_1 and alpha_2 differ by a lot this may skew the effect of pheromones
        self.discount_1 = float(self.graph.gamma) / float(self.graph.alpha_1)
        self.discount_2 = float(self.graph.gamma) / float(self.graph.alpha_2)
        
        self.tau0 = params["tau0"]
        self.q0 = params["q0"]
        self.alpha = params["alpha"]
        self.beta = params["beta"]

        # pheromone matrices based on ordered edges (tuples) 
        self.pheromones_1 = []
        self.pheromones_2 = []

        for v1,v2 in self.graph.graph.edges():
            edge = utils.ordered_edge(v1,v2)
            self.pheromones_1[edge] = self.tau0
            self.pheromones_2[edge] = self.tau0


class Ant:
    def __init__(self, aco : AntColonyOptimization):
        self.aco = aco
        
        # terminals yet to be traversed
        self.terminals_1 = self.aco.graph.terminal_1.copy()
        self.terminals_2 = self.aco.graph.terminal_2.copy()

        # start terminals
        n1 = self.terminals_1[random.randrange(0, len(self.terminals_1))]
        n2 = self.terminals_2[random.randrange(0, len(self.terminals_2))]
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
        
        # candidate edges
        self.cand_1 = {utils.ordered_edge(n1, n) for n in self.aco.graph.graph.neighbors(n1)}
        self.cand_2 = {utils.ordered_edge(n2, n) for n in self.aco.graph.graph.neighbors(n2)}
    
    
    def done(self):
        return len(self.terminals_1) + len(self.terminals_2) == 0
    
    
    def take_step(self):
        if len(self.terminals_1) > 0:        
            # pick an edge
            ranking = map(lambda e: compute_edge_objective(e, self.aco.pheromones_1, self.edges_2, self.aco.discount_1), self.cand_1)
            q = random.random()
            if q < self.aco.q0:
                # clever fast way to get the maximum index (see https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list)
                index = max(range(0, len(ranking)), key=ranking.__getitem__)
                edge = self.cand_1[index]
            else:
                edge = random.choices(self.cand_1, weights=ranking)

            # add edge
            self.edges1.add(edge)
            if edge in self.edges_2:
                self.edges_shared.add(edge)

            self.weight_1 += self.aco.graph.graph.get_edge_data(*edge)["weight_1"]
                        
            # update data structures
            new_node = edge[0] if edge[0] in self.nodes_1 else edge[1]
            self.nodes_1[edge[0]] = self.nodes_1.get(edge[0], 0) + 1
            self.nodes_1[edge[1]] = self.nodes_1.get(edge[1], 0) + 1
            
            self.terminals_1.discard(new_node)
            
            cand_add = {utils.ordered_edge(new_node, n) for n in self.aco.graph.graph.neighbors(new_node)}
            self.cand_1.update(cand_add)
            cand_del = {e for e in self.cand_1 if e[0] in self.nodes_1 and e[1] in self.nodes_1}
            self.cand_1.difference_update(cand_del)

        if len(self.terminals_2) > 0:        
            # pick an edge
            ranking = map(lambda e: compute_edge_objective(e, self.aco.pheromones_2, self.edges_1, self.aco.discount_2), self.cand_2)
            q = random.random()
            if q < self.aco.q0:
                # clever fast way to get the maximum index (see https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list)
                index = max(range(0, len(ranking)), key=ranking.__getitem__)
                edge = self.cand_2[index]
            else:
                edge = random.choices(self.cand_2, weights=ranking)

            # add edge
            self.edges2.add(edge)
            if edge in self.edges_1:
                self.edges_shared.add(edge)

            self.weight_2 += self.aco.graph.graph.get_edge_data(*edge)["weight_2"]
                        
            # update data structures
            new_node = edge[0] if edge[0] in self.nodes_2 else edge[1]
            self.nodes_2[edge[0]] = self.nodes_2.get(edge[0], 0) + 1
            self.nodes_2[edge[1]] = self.nodes_2.get(edge[1], 0) + 1
            
            self.terminals_2.discard(new_node)
            
            cand_add = {utils.ordered_edge(new_node, n) for n in self.aco.graph.graph.neighbors(new_node)}
            self.cand_2.update(cand_add)
            cand_del = {e for e in self.cand_2 if e[0] in self.nodes_2 and e[1] in self.nodes_2}
            self.cand_2.difference_update(cand_del)
            
    
    def compute_edge_objective(edge, pheromones, discounted_edges, discount):
        f1 = pheromones[edge] ** self.aco.alpha
        if edge in discounted_edges:
            f2 = ((1.0 / (self.aco.graph.graph.get_edge_data(*e)["weight"] - discount)) ** self.aco.beta)
        else:
            f2 = ((1.0 / self.aco.graph.graph.get_edge_data(*e)["weight"]) ** self.aco.beta)
        return f1 * f2
    
    
    def get_solution(self):
        pass