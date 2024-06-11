import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym
from gymnasium import spaces



import numpy as np





def expand_matrix(matrix):
    # Get the number of rows (n) and columns (n-1) of the input matrix
    n, n_minus_1 = matrix.shape
    
    # Create an n x n matrix filled with zeros
    expanded_matrix = np.zeros((n, n))
    
    # Insert the elements from the input matrix into the new matrix
    for i in range(n):
        expanded_matrix[i, :i] = matrix[i, :i]
        expanded_matrix[i, i+1:] = matrix[i, i:]
    
    return expanded_matrix



b = np.array([[1, 2], [3, 4], [5, 6]])
a = expand_matrix(b)
print(a)




""" def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

mixing_matrix = np.array([[0.0, 1000], [0.3, 0.1], [0.5, 0.5]])
a=softmax(mixing_matrix[0])
print(a) """





""" a= [{'test_acc': [5,4,3], 'TimeLimit.truncated': False}]  
b= a[0]['test_acc']
print(b)
 """

""" class A:
     def __init__(self):
        self.acc = []

    
     def method(self):
         a = 1
         b = 2
         self.acc.append(a + b)


b = A()
for i in range(10):
    b.method()
     """










""" class b:
    def __init__(self, seed=None):
        self.seed = seed
        self.random_state = np.random.RandomState(seed) if seed is not None else np.random.RandomState()


    def generate_graph(self, num_agents, graph_connectivity):
        np.random.seed(self.seed)  # Set seed for reproducibility
        while True:
            graph = nx.random_geometric_graph(num_agents, graph_connectivity)
            if nx.is_k_edge_connected(graph, 1):
                break
        nx.draw(graph, with_labels=True)
        plt.show()
        return graph

a = b(20)
a.generate_graph(10, 0.2) """

# Example list of floats
""" float_list = [1.1, 2.2, 3.3]
obs = [0.0]*10

# Convert list of floats to NumPy array
array = np.array(obs)

print(array)
print(type(array))
print(array.shape)
 """
""" class YourClass:
    def __init__(self):
        self.num_agents = 10  # Number of agents
        self.graph_connectivity = 0.2  # Connectivity parameter for random geometric graph
        # self.node_labels = list(range(self.num_agents))  # List of node labels or IDs
        self.graph = None
        self.seed = 42
        

    def generate_graph(self):
        while True:            
            # Create the graph with specified node labels
            graph = nx.random_geometric_graph(self.num_agents, self.graph_connectivity, seed=self.seed)
            # nx.relabel_nodes(graph, dict(enumerate(self.node_labels)), copy=False)
            # Ensure the graph is connected
            
            if nx.is_k_edge_connected(graph, 1):
                # Assign node labels to the generated graph
                # nx.relabel_nodes(graph, dict(enumerate(self.node_labels)), copy=False)
                return graph
            else:
                self.seed = self.seed + 1

   

# Usage
ins= YourClass()
for i in range(3):
     a=ins.generate_graph()
     nx.draw_circular(a, with_labels=True)
     plt.show()



b = 1 """
