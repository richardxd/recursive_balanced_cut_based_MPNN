import numpy as np
import networkx as nx


def get_reverse_and_prefix_sum(
        G: nx.Graph, indices: np.ndarray) -> [np.ndarray, np.ndarray]:
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


def get_cut_weights(G: nx.Graph, indices: np.ndarray) -> np.ndarray:
    cut_weights = np.zeros(len(indices))
    prefix_sum, reversed_sum = get_reverse_and_prefix_sum(G, indices)
    cut_weights[0] = reversed_sum[0]
    for i in range(1, len(indices)):
        cut_weights[i] = cut_weights[i - 1] - prefix_sum[i] + reversed_sum[i]
    return cut_weights


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
    for u, v in G.edges():
        # assuming there is no duplicate edges
        weighted_degree[u] += G[u][v]['weight']
        weighted_degree[v] += G[u][v]['weight']

    vol_sum[0] = weighted_degree[indices[0]]
    for i in range(1, len(indices)):
        vol_sum[i] = vol_sum[i - 1] + weighted_degree[indices[i]]
    return vol_sum


def sweep_cut(G: nx.Graph,
              x: np.ndarray,
              lower_vol: float = np.inf,
              upper_vol: float = np.inf,
              reversed_cut: bool = True,
              partial: float | None = None) -> (float, float, np.ndarray):
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

    cut_weights = get_cut_weights(G, sorted_idx)
    vol_sum = get_volume_prefix(G, sorted_idx)

    conductance_arr = cut_weights[:-1] / np.minimum(
        vol_sum, vol_sum[-1] - vol_sum)[:-1]  #

    # Mask conductance values outside the volume bounds with infinity
    # right now S_i = {j in [n] | x_j <= x_i}
    # the volume of S_i is vol_sum[i] if not reversed
    # otherwise, compute the volume of S_i by vol_sum[-1] - vol_sum[i]
    vol = vol_sum[-1] - vol_sum if reversed_cut else vol_sum
    # print(vol.shape)
    # print(conductance_arr.shape)
    conductance_arr[(vol < lower_vol)[:-1] | (vol > upper_vol)[:-1]] = np.inf

    if partial:
        # run sweep cut on G with embedding r. Let z be the smallest index such that vol(S_z) >= b/4 * 2m
        # output the most balanced sweep cut among {S1, ..., Sz-1} such that the conductance is at most 40 \sqrt{gamma}
        # if no such cut exists, raise an error

        # find the smallest inde such that vol(S_z) >= b/4 * 2m
        # vol_sum[-1] is the total volume of the cut
        z = np.searchsorted(vol_sum, partial)
        if z == n:
            raise ValueError(f'Partial volume {partial} is too large.')
        conductance_arr = conductance_arr[:z]

    if len(x.shape) != 1:
        raise ValueError(
            f'Embedding should be one dimensional. Instead got `x` with shape {x.shape}.'
        )

    cut = np.zeros(n, int)

    min_idx = np.argmin(conductance_arr)
    thr = x[sorted_idx[min_idx]]
    conductance = conductance_arr[min_idx]

    cut = (x <= thr).astype(int)
    if reversed_cut:
        cut = 1 - cut

    return thr, conductance, cut
