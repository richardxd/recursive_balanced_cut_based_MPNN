import unittest
import networkx as nx
from balanced_separator import BalSep

class TestBalSep(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_edge(1, 2, weight=1)
        self.graph.add_edge(2, 3, weight=1)
        self.bal_sep = BalSep(self.graph)

    def test_balanced_separator(self):
        b = 0.3
        gamma = 1 
        epsilon = 1
        alpha = 1
        result = self.bal_sep.balanced_separator(b, gamma, epsilon, alpha)
        print(result)
        # Use assert statements to verify the result
        # Replace this with the actual expected result
    
    

if __name__ == '__main__':
    unittest.main()