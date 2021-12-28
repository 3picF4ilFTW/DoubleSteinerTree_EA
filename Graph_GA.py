import networkx as nx
import utils
import itertools
import matplotlib.pyplot as plt



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
                edges.append(utils.ordered_edge_weighted(int(line[1]), int(line[2]),
                                                         int(line[3])))  # collects edges out of the file with weight
            if line[0] == "T1":
                self.terminal_1.append(int(line[1]))  # collects the Terminals1 out of file
            if line[0] == "T2":
                self.terminal_2.append(int(line[1]))

        self.graph.add_nodes_from(vertices)
        self.graph.add_weighted_edges_from(edges)

        for n1, n2, d in self.graph.edges.data():
            d["weight_1"] = self.alpha_1 * d["weight"]
            d["weight_2"] = self.alpha_2 * d["weight"]
            d["weight_modifier"] = self.gamma * d["weight"]
            d["weight_1_mod"] = self.alpha_1 * d["weight"]
            d["weight_2_mod"] = self.alpha_2 * d["weight"]

    def reset_edge_weights(self, treeind: int):
        for n1, n2, d in self.graph.edges.data():
            d[f"weight_{treeind}_mod"] = getattr(self, f"alpha_{treeind}") * d["weight"]

    def verify_correctness(self, steiner_tree1: nx.Graph, steiner_tree2: nx.Graph, verbose = 0): #input all nodes of the steiner tree

        if nx.components.is_connected(steiner_tree1) and nx.components.is_connected(steiner_tree2):
            for t1 in self.terminal_1:
                if not t1 in steiner_tree1.nodes:
                    if verbose >0:
                        print(f"{t1} is not in the steiner tree 1")
                    return False
            for t2 in self.terminal_2:
                if not t2 in steiner_tree2.nodes:
                    if verbose > 0:
                        print(f"{t2} is not in the steiner tree 2")
                    return False
        else:
            if verbose > 0:
                print("One of the steiner trees is not connected")
            return False
        return True
    """
        Builds the distance graph using the nodes in node list
        For the weights the variables weight_{tree}_{mod} are used
        this means we can built the distance graph with whatever weight of the tree we like
    """
    #TODO in the distance graph function we can have the biggest time safe if we figure out how to reuse them or are able to not build them at all
    def distance_graph(self, node_list: [int], tree: int, mod : bool):
        graph = nx.Graph()
        graph.add_nodes_from(node_list)
        if not mod:
            for s, t in itertools.combinations(node_list, 2):
                length, path = nx.algorithms.shortest_paths.weighted.bidirectional_dijkstra(self.graph, s, t, f"weight_{tree}")
                graph.add_edge(s, t, weight=length, path=path)
        else:
            for s, t in itertools.combinations(node_list, 2):
                length, path = nx.algorithms.shortest_paths.weighted.bidirectional_dijkstra(self.graph, s, t,
                                                                                            f"weight_{tree}_mod")
                graph.add_edge(s, t, weight=length, path=path)

        return graph

    def augment_distance_graph(self, dg: nx.Graph, n: int, weight: str, already_solved_dict: dict = None):
        if already_solved_dict is None:
            dg.add_node(n)
            for s in dg.nodes():
                if s == n:
                    continue
                length, path = nx.algorithms.shortest_paths.weighted.bidirectional_dijkstra(self.graph, s, n, weight)
                dg.add_edge(s, n, weight=length, path=path)
        else:
            dg.add_node(n)
            for s in dg.nodes():
                if s == n:
                    continue
                if (s,n) in already_solved_dict:
                    length, path = already_solved_dict[(s,n)][0], already_solved_dict[(s,n)][1]
                else:
                    length, path = nx.algorithms.shortest_paths.weighted.bidirectional_dijkstra(self.graph, s, n, weight)
                    already_solved_dict[(s,n)] = (length, path)

                dg.add_edge(s, n, weight=length, path=path)
    """
        Builds the steiner tree from the distnace graph built in the function above
        Uses the weights in the distance graph for calculating the mst.
        The tree variable determines which terminal nodes are used.
        The tree variable also determines which weights from the graph are used.
    """
    def rebuild_steiner_tree(self, dg: nx.Graph, tree: int):
        mst = nx.algorithms.tree.mst.minimum_spanning_tree(dg, weight=f"weight")

        edges = set()

        # remove unnecessary nodes
        terminals = getattr(self, f"terminal_{tree}")
        changed = True
        while changed:
            changed = False
            to_remove = []

            for n in mst.nodes:
                if n not in terminals and mst.degree(n) == 1:
                    to_remove.append(n)
                    changed = True

            mst.remove_nodes_from(to_remove)

        # translate distance graph edges to original graph edges
        edges_weighted = set()
        for (n1, n2) in mst.edges():
            last_n = None
            for n in dg.get_edge_data(n1, n2)["path"]:
                if last_n is None:
                    last_n = n
                    continue
                w = self.graph.get_edge_data(last_n, n)
                w = w[f"weight_{tree}"]
                edges.add(utils.ordered_edge(last_n, n))
                edges_weighted.add(utils.ordered_edge_weighted(last_n, n, w))
                last_n = n

        #   For cycle detection build the graph and build the mst unfortunately this is the fastest I could think of for cycle detection.
        #   Because if we use the cycle function of nx we still need to figure out which path is the best to delete which seems like a very daunting task
        #   Therefore cycle deletion is just build the mst and then remove nodes of degree 1 that arent terminal nodes.
        nodes = set()
        graph = nx.Graph()
        for n1, n2 in edges:
            nodes.add(n1)
            nodes.add(n2)
        graph.add_nodes_from(nodes)
        graph.add_weighted_edges_from(edges_weighted)
        # to remove cycles build the mst of the supposed steiner tree
        st_mst = nx.minimum_spanning_tree(graph, "weight")
        if st_mst.edges != graph.edges:
            # if the edges changed at least one cycle was removed therefore we need to trim the tree further
            # we do the trimming as before
            changed = True
            while changed:
                changed = False
                to_remove = []

                for n in st_mst.nodes:
                    if n not in terminals and st_mst.degree(n) == 1:
                        to_remove.append(n)
                        changed = True

                st_mst.remove_nodes_from(to_remove)

        # We could have introduced key-nodes which are not in the MST! Therefore, we have to recompute the key-nodes!
        # Also compute the weight of the new tree.
        node_degree = {}
        total_weight = 0
        edges = set()
        weighted_edges = []
        for n1, n2 in st_mst.edges:
            if n1 not in node_degree:
                node_degree[n1] = 0
            if n2 not in node_degree:
                node_degree[n2] = 0
            node_degree[n1] += 1
            node_degree[n2] += 1

            total_weight += self.graph.get_edge_data(n1, n2)[f"weight_{tree}"]

            edges.add(utils.ordered_edge(n1, n2))
            weighted_edges.append(utils.ordered_edge_weighted(n1, n2, self.graph.get_edge_data(n1, n2)[f"weight_{tree}"]))
        steiner_tree = nx.Graph()
        steiner_tree.add_nodes_from(nodes)
        steiner_tree.add_weighted_edges_from(weighted_edges)
        #self.modify_weights(steiner_tree, tree)
        return steiner_tree

    def modify_weights(self,steiner_tree: nx.Graph, tree_index):#give the steiner tree and the index of the steiner tree
        for edge in steiner_tree.edges:
            d = self.graph.get_edge_data(edge[0],edge[1])
            #we need to modify the weight of the other tree after a tree has been built
            #tree index is either 1 or 2 we need to modify the other tree thus 3-tree_index
            d[f"weight_{3-tree_index}_mod"] = d[f"weight_{3-tree_index}"] + d["weight_modifier"]



    def draw_graph(self, g):
        nx.draw(g, with_labels=True)
        plt.draw()
        plt.show()