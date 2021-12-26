from Graph import Graph

class AntSolution:
    def __init__(self):
        self.edges_1 = set()
        self.edges_2 = set()
        
        self.weight = None
    
    
    def __str__(self):
        return f"E1: {self.edges_1}\nE2: {self.edges_2}\nObj: {self.weight}"
        
        
    def write_solution_to_file(self, g : Graph, file : str):
        f = open(file, "w")
        
        f.write(f"S {g.abs_V} {g.abs_E} {g.abs_T_1} {g.abs_T_2}\n")
        
        for n1, n2 in self.edges_1:
            f.write(f"S1 {n1} {n2}\n")
            
        for n1, n2 in self.edges_2:
            f.write(f"S2 {n1} {n2}\n")
        
        f.close()
    