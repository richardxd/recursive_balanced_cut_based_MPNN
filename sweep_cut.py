from typing import Tuple, Dict, Hashable

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



def get_reverse_and_prefix_sum(G: nx.Graph, x: np.ndarray, indices : np.ndarray) -> [np.ndarray, np.ndarray]:
    '''
    Given the sorted indices of the node embedding, return the reversed sum of the edge weights, where
        prefix_sum[i] is defined as the sum of the edge weights of nodes
        indices[j] with j < i.
        reversed_sum[i] is defined as the sum of the edge weights of nodes indices[j] with j > i. 
    return prefix_sum, reversed_sum
    '''
    reversed_sum = np.zeros(len(indices))
    prefix_sum = np.zeros(len(indices))
    indices_index = {indices[i]: i for i in range(len(indices))}
    for u, v in G.edges():
        u_idx = indices_index[u] 
        v_idx = indices_index[v] 
        if u_idx > v_idx:
            prefix_sum[u_idx] += G[u][v]['weight']
            reversed_sum[v_idx] += G[u][v]['weight']
        elif u_idx < v_idx:
            prefix_sum[v_idx] += G[u][v]['weight']
            reversed_sum[u_idx] += G[u][v]['weight']
        # else:
        #     raise Exception('Self loop is not allowed')
    return prefix_sum, reversed_sum

def get_cut_weights(G: nx.Graph, x: np.ndarray, indices : np.ndarray) -> np.ndarray:
    cut_weights = np.zeros(len(indices))
    prefix_sum, reversed_sum = get_reverse_and_prefix_sum(G, x, indices)
    cut_weights[0] = reversed_sum[0] 
    for i in range(1, len(indices)):
        cut_weights[i] = cut_weights[i-1] - prefix_sum[i] + reversed_sum[i]
    return cut_weights


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


# TODO: assemble the test functions into one single test function
 
def get_volume_prefix(G: nx.Graph, indices: np.ndarray) -> np.ndarray:
    '''
    compute the prefix sum of volume with respect to indices
    vol_sum[i] is the sum of weights of all vertex degree of nodes indices[j] with j < i
    '''
    vol_sum = np.zeros(len(G))
    indices_index = {indices[i]: i for i in range(len(indices))}
    # compute the edge degree
    weighted_degree = np.zeros(len(G))
    for u,v in G.edges():
        # assuming there is no duplicate edges
        weighted_degree[u] += G[u][v]['weight']
        weighted_degree[v] += G[u][v]['weight']
    
    vol_sum[0] = weighted_degree[indices[0]]
    for i in range(1, len(indices)):
        vol_sum[i] = vol_sum[i-1] + weighted_degree[indices[i]]
    return vol_sum

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

def sweep_cut(G: nx.Graph, x: np.ndarray, lower_vol:np.float('-inf'), upper_vol:float=np.float('inf'), reversed_cut:bool=True, partial:float | None = None) -> (float, float, np.ndarray):
    """Sweep Cut
    
    Also known as threshold cut, takes in a graph and a
    1D embedding of the nodes and returns a threshold,
    the conductance from the associated threshold cut and
    a 0/1 vector indicating which side of the cut each node
    belongs to in the best conductance cut.
    
    Parameters
    ----------
    G : nx.Graph
        The graph to compute the threshold cut for.
    x : np.ndarray
        1D Embedding of the nodes in G.
    lower_vol: float
        The lower bound of the threshold for the acceptable volume for a cut
    upper_vol: float
        The upper bound of the threshold for the acceptable volume for a cut
        
    Returns
    -------
    thr : float
        The best conductance threshold.
    conductance : float
        Conductance \phi(S) = w(E(S, \bar{S})) / min(vol(S), vol(\bar{S}))
        of the best threshold cut
    cut : np.ndarray
        0/1 indicator vector of the best threshold cut.
        It holds that cut = (x <= thr)
    """
    n = len(G)
    # TODO: Compute the threshold for best conductance cut
    #
    #       \phi(S) = w(E(S, \bar{S})) / min(vol(S), vol(\bar{S}))
    #
    #       Your code should take advantage of the fact that
    #       the difference between two adjacent threshold cuts
    #       is a single node. Total running time should be O(|V| + |E|).
    
    # sort the node
    sorted_idx = np.argsort(x)
    # idx_list = list(range(n))




    cut_weights = get_cut_weights(G, x, sorted_idx)
    vol_sum = get_volume_prefix(G, sorted_idx) 

    conductance_arr = cut_weights[:-1] / np.minimum(vol_sum, vol_sum[-1] - vol_sum)[:-1] # 

    # Mask conductance values outside the volume bounds with infinity
    # right now S_i = {j in [n] | x_j <= x_i}
    # the volume of S_i is vol_sum[i] if not reversed
    # otherwise, compute the volume of S_i by vol_sum[-1] - vol_sum[i]
    vol = vol_sum[-1] - vol_sum if reversed_cut else vol_sum 
    conductance_arr[(vol < lower_vol) | (vol > upper_vol)] = np.inf

    if partial:
        # run sweep cut on G with embedding r. Let z be the smallest index such that vol(S_z) >= b/4 * 2m
        # output the most balanced sweep cut among {S1, ..., Sz-1} such that the conductance is at most 40 \sqrt{gamma}
        # if no such cut exists, raise an error
        
        # find the smallest index such that vol(S_z) >= b/4 * 2m
        # vol_sum[-1] is the total volume of the cut
        z = np.searchsorted(vol_sum, partial)
        if z == n:
            raise ValueError(f'Partial volume {partial} is too large.')
        conductance_arr = conductance_arr[:z]

    if len(x.shape) != 1:
        raise ValueError(f'Embedding should be one dimensional. Instead got `x` with shape {x.shape}.')
    
    cut = np.zeros(n, int)
    
    min_idx = np.argmin(conductance_arr)
    thr = x[sorted_idx[min_idx]]
    conductance = conductance_arr[min_idx]

    cut =  (x <= thr).astype(int)
    if reversed_cut:
        cut = 1 - cut
    
    
    return thr, conductance, cut


n = 100
P = nx.path_graph(n)
nx.set_edge_attributes(P, 1, 'weight')
x = np.linspace(0, 1, n)
thr, cond, cut = sweep_cut(P, x)

print(f'On the path with {n} nodes the best conductance threshold is thr = {thr:.6f} and yields conductance {cond:.6f}.')
pos = nx.spectral_layout(P)
nx.draw(P, pos=pos, node_color=colors[cut], node_size=50)
    