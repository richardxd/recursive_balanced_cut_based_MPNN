import numpy as np
import unittest
import networkx as nx
from balanced_separator import BalSep, get_degree_vector
import matplotlib.pyplot as plt


class TestBalSep(unittest.TestCase):

    def setUp(self):
        self.graph = nx.Graph()
        self.b = 0.2
        self.gamma = 1
        self.epsilon = 1
        self.alpha = 3

    # in general, beta should be below 0.2, 0.3
    def test_balanced_separator_simple(self):
        self.graph = nx.Graph()
        self.graph.add_edge(1, 2, weight=1)
        self.graph.add_edge(2, 3, weight=1)
        self.bal_sep = BalSep(self.graph)
        result = self.bal_sep.balanced_separator(self.b, self.gamma,
                                                 self.epsilon, self.alpha)
        
        print(result)

        # check if the result is b-balanced
        if result is not None:
            assert self.check_balanced(self.graph, result, self.b)

        # Use assert statements to verify the result
        # Replace this with the actual expected result

    def test_balanced_separator_simple2(self):
        self.graph = nx.Graph()
        self.graph.add_edge(1, 2, weight=1)
        self.graph.add_edge(2, 3, weight=1)
        self.graph.add_edge(1, 3, weight=1)
        self.bal_sep = BalSep(self.graph)
        self.b = 0.3
        result = self.bal_sep.balanced_separator(self.b, self.gamma,
                                                 self.epsilon, self.alpha)

        print(result)

        if result is not None:
            assert self.check_balanced(self.graph, result, self.b)

    def test_balanced_separator_path(self):
        self.graph = nx.path_graph(10)
        for edges in self.graph.edges():
            self.graph[edges[0]][edges[1]]['weight'] = 1
        self.bal_sep = BalSep(self.graph)
        result = self.bal_sep.balanced_separator(self.b, self.gamma,
                                                 self.epsilon, self.alpha)
        print(result)

        if result is not None:
            assert self.check_balanced(self.graph, result, self.b)

    # def test_balanced_separator_bolus(self):
    #     self.graph = nx.Graph()

    def visualize_graph(self, cut: np.ndarray):
        # draw the graph with a given cut in one hot manner. Color the nodes in the cut red
        color = ['red' if i == 1 else 'blue' for i in cut]
        nx.draw(self.graph, node_color=color, with_labels=True)
        plt.show()

    def check_balanced(self, graph: nx.Graph, cut: np.ndarray,
                       b: float) -> bool:
        '''
        Input:
            graph: an unweighted instance graph G = (V,E)
            cut: a np.ndarray of length n, where cut[i] = 1 if i is in the cut, 0 otherwise
            b: a constant balance value b \in (0, 1/2]
        Output:
            True if the cut is balanced, False otherwise
        '''

        # compute the degree vector of the graph
        degree_vector = get_degree_vector(graph)
        volume_cut = np.dot(degree_vector, cut)
        # find the complement of the cut
        volume_graph = sum(degree_vector)
        volume_complement = volume_graph - volume_cut
        if min(volume_cut, volume_complement) >= b * volume_graph:
            return True
        return False


if __name__ == '__main__':
    unittest.main()
