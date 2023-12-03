import matplotlib as mpl
import matplotlib.pyplot as plt
from sweep_cut import *
# Path: test.py

# Test case 1
# Create a small graph
G = nx.Graph()
G.add_edge(0, 1, weight=2)
G.add_edge(0, 2, weight=3)
G.add_edge(1, 2, weight=4)

# Define the node embeddings
x = np.array([0.5, 0.2, 0.8])

# Define the sorted indices
indices = np.argsort(x)

# Expected results
expected_prefix_sum = np.array([0, 2, 7], dtype=np.float64)
expected_reversed_sum = np.array([6, 3, 0], dtype=np.float64)
expected_cut_weights = np.array([6, 7, 0], dtype=np.float64)

# Test get_reverse_and_prefix_sum function
prefix_sum, reversed_sum = get_reverse_and_prefix_sum(G, x, indices)

assert np.array_equal(prefix_sum, expected_prefix_sum)
assert np.array_equal(reversed_sum, expected_reversed_sum)

# Test get_cut_weights function
cut_weights = get_cut_weights(G, x, indices)
assert np.array_equal(cut_weights, expected_cut_weights)

print("Test cases passed!")

# Create a small graph
G = nx.Graph()
G.add_edge(0, 1, weight=2)
G.add_edge(0, 2, weight=3)
G.add_edge(1, 2, weight=4)

# Define the node embeddings
x = np.array([0.5, 0.2, 0.8])

# Define the sorted indices
indices = np.argsort(x)

# Compute the volume of the cut
vol_sum = get_volume_prefix(G, indices)

# Expected results
expected_vol_sum = np.array([6, 11, 18], dtype=np.float64)

# Check if the computed volume of the cut matches the expected results
assert np.array_equal(vol_sum, expected_vol_sum)

print("Test case passed!")

# TODO: assemble the test function into on

n = 100
P = nx.path_graph(n)
nx.set_edge_attributes(P, 1, 'weight')
x = np.linspace(0, 1, n)
thr, cond, cut = sweep_cut(P, x)

print(
    f'On the path with {n} nodes the best conductance threshold is thr = {thr:.6f} and yields conductance {cond:.6f}.'
)
pos = nx.spectral_layout(P)
nx.draw(P, pos=pos, node_color=colors[cut], node_size=50)

if __name__ == "__main__":
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])
