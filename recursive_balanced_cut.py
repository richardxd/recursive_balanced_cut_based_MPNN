import numpy as np
import networkx as nx
from scipy.sparse.csgraph import laplacian
import scipy

from sweep_cut import sweep_cut, get_volume_prefix

from typing import Tuple, Dict, Hashable

import matplotlib as mpl
import matplotlib.pyplot as plt

colors = np.array(['tab:blue', 'orange'])


def random_unit_vectors(k, n):
    # Generate k random vectors in R^n
    vectors = np.random.randn(k, n)
    # Normalize each vector to have unit length
    unit_vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    return unit_vectors


def project_embedding(vectors, heat_kernel):
    # Placeholder for projection step
    # return np.dot(vectors, heat_kernel)
    pass


def find_cut(graph, vectors):
    # Placeholder for FindCut subroutine
    # This would be a complex function to implement, involving SDP and sweep cuts
    # return set(), False  # Return a cut and a boolean indicating if it's balanced
    pass


def expv(vectors):
    # Placeholder for ExpV subroutine
    # return vectors
    pass


# Code below defines some of the preliminary quantities and special matrices
# section 5.1, basic preliminaries:
## instance graph and edge volume


def get_volume(graph: nx.graph, cut: np.ndarray) -> int:
    '''
    Input:
        graph: an unweighted instance graph G = (V,E)
        cut: a np.ndarray of length n, where cut[i] = 1 if i is in the cut, 0 otherwise 
    Output:
        The volume of the cut
    '''

    degree_vector = nx.degree_matrix(graph, weight='weight')
    # for i in cut:
    #     volume += graph.degree(i, weight='weight')
    volume = np.dot(degree_vector, cut)
    return volume


def check_balanced(graph: nx.graph, cut: np.ndarray, b: float) -> bool:
    '''
    Input:
        graph: an unweighted instance graph G = (V,E)
        cut: a np.ndarray of length n, where cut[i] = 1 if i is in the cut, 0 otherwise 
        b: a constant balance value b \in (0, 1/2]
    Output:
        True if the cut is balanced, False otherwise
    '''

    volume = get_volume(graph, cut)
    # find the complement of the cut
    cut_complement = 1 - cut
    volume_complement = get_volume(graph, cut_complement)
    volume_v = graph.get_volume(graph, [1] * len(graph))
    if min(volume, cut_complement) >= b * volume_v:
        return True
    return False


## speical graph
def complete_graph(graph) -> nx.Graph:
    '''
    Input:
        graph: an unweighted instance graph G = (V,E)
    Output:
        A complete graph with weight d_i d_j / 2m between every pair i,j in V.
    '''

    complete_graph = nx.Graph()
    m = graph.number_of_edges()
    n = graph.number_of_nodes()
    for i in graph.nodes():
        for j in graph.nodes():
            if i != j:
                d_i = graph.degree(i, weight='weight')
                d_j = graph.degree(j, weight='weight')
                complete_graph.add_edge(i, j, weight=d_i * d_j / (2 * m))
    return complete_graph


def star_graph(graph, v) -> nx.Graph:
    '''
    Input:
        graph: an unweighted instance graph G = (V,E)
        v: a vertex in V
    Output:
        S_i: the star grpah rooted at i, with edge weight of d_id_j/2m between i and j for all j in V
    '''
    star_graph = nx.Graph()
    for j in graph.nodes():
        if j != v:
            d_i = graph.degree(v, weight='weight')
            d_j = graph.degree(j, weight='weight')
            star_graph.add_edge(v, j, weight=d_i * d_j / (2 * m))
    return star_graph


## Fact 5.1
def compute_laplacian_complete_graph(graph) -> np.ndarray:
    '''
    Input:
        graph: an unweighted instance graph G = (V,E)
    Output:
        laplacian of the special complete graph obtained via the function complete_graph()
    '''

    complete_graph = complete_graph(graph)
    D = nx.degree_matrix(complete_graph, weight='weight')
    m = complete_graph.number_of_edges()
    ones = np.ones(n)
    L = D - 1 / (2 * m) * D @ ones @ ones.T @ D
    return L


## Embedding Notation, v_avg
def compute_average_node_embedding(graph, embeddings) -> np.ndarray:
    '''
    Input:
        embeddings: a list of n-dimensional vectors, where each vector is an embedding of the node v_i
    Output:
        the mean vector of the embeddings defined as
        v_avg := sum_{i in V} d_i/(2m) * v_i
    '''
    v_avg = np.zeros(len(embeddings[0]))
    m = graph.number_of_edges()
    for i in range(len(embeddings)):
        v_avg += embeddings[i] * graph.degree(i, weight='weight') / (2 * m)
    return v_avg


# Section 5.2, AHK random walk
## Denote P_tau (beta) as:
## Probability Transitimation Matrix of H(beta) between time 0 and tau
## P_tau (beta) = exp(tau Q(beta))
## where
## Q(beta) = -( L + sum_i in V beta_i L(S_i)) D^-1


def compute_transition_rate_matrix(graph, L, beta,
                                   degree_vector) -> np.ndarray:
    '''
    Input:
        L: the Laplacian matrix of the instance graph
        beta: a vector of length n
        degree_vector one dimension vector of the diagonal of the degree matrix of the instance graph 
    Output:
        Q(beta) = -( L + sum_i in V beta_i L(S_i)) D^-1
    '''
    n = len(L)
    D_inv = np.diag(1 / degree_vector)
    Q = L
    laplacian_star_graphs = [
        star_graph(graph, i).laplacian_matrix for i in range(graph.nodes())
    ]

    for i in range(n):
        Q += beta[i] * laplacian_star_graphs[i]
    Q = -Q @ np.linalg.inv(D)
    return Q


def compute_transition_matrix(graph, L, beta, degree_vector,
                              tau) -> np.ndarray:
    '''
    Input:
        L: the Laplacian matrix of the instance graph
        beta: a vector of length n
        degree_vector one dimension vector of the diagonal of the degree matrix of the instance graph 
        tau: time parameter
    Output:
        P_tau (beta) = exp(tau Q(beta)) 
    '''
    Q = compute_transition_rate_matrix(graph, L, beta, degree_vector)
    return scipy.linalg.expm(tau * Q)


## Mixing:


## psi(P_tau(beta), i) := d_i sum_{j in V} d_j ((e_j^T P_tau(beta) e_i)/d_j   - 1/2m)^2
def compute_mixing(graph, tau, beta, i) -> float:
    '''
    Input:
        graph: an unweighted instance graph G = (V,E)
        tau: time parameter
        beta: a vector of length n
        i: a vertex in V
    Output:
        psi(P, i) := d_i sum_{j in V} d_j ((e_j^T P e_i)/d_j   - 1/2m)^2
    '''
    # brute force

    # m = graph.number_of_edges()
    # d_i = graph.degree(i, weight='weight')
    # sum_ = 0
    # for j in graph.nodes():
    #     d_j = graph.degree(j, weight='weight')
    #     sum_ += d_j * ((P[j][i] / d_j) - 1/(2 * m))**2
    # ------------------
    # alternatively, following from fact 5.4:
    # psi(P, i) = d_i R_i \bullet D^-1 P_2tau(beta)
    # degree_vector = nx.degree_matrix(graph, weight='weight')
    # P_2tau_beta = compute_transition_matrix(graph, L, beta, degree_vector, 2 * tau)

    # compute R_i
    # return d_i * sum_
    pass


## deviation of a cut
## psi(P_tau(beta), S) := sum_{i in S} psi(P_tau(beta), i)
def compute_total_deviation(graph, P, S) -> float:
    '''
    Input:
        P: a probability transition matrix
        S: a subset of V
    Output:
        psi(P, S) := sum_{i in S} psi(P, i) 
    '''
    pass
    # sum_ = 0
    # for i in S:
    # sum_ += compute_mixing(graph, P, i)
    # return sum_


## total deviation of a graph:
## psi(P_tau(beta), G) = L(K_V) \bullet D^-1 P_2tau(beta)
def compute_total_deviation_graph(graph, tau, beta) -> float:
    '''
    Input:
        P: a probability transition matrix
    Output:
        psi(P, G) = L(K_V) \bullet D^-1 P_2tau(beta)
    '''
    degree_vector = nx.degree_matrix(graph, weight='weight')
    D_inv = np.diag(1 / degree_vector)
    L = nx.laplacian_matrix(graph, weight="weight")
    P_2tau_beta = compute_transition_matrix(graph, L, beta, degree_vector,
                                            2 * tau)
    L_K_V = compute_laplacian_complete_graph(graph)

    return np.trace(L_K_V @ D_inv) @ P_2tau_beta


def update_beta(beta, cut, gamma, T, S) -> np.ndarray:
    '''
    Input:
        beta: a vector of length n
        cut: a subset of V
        t: current iteration
        T: total number of iterations
        S: a subset of V
    Output:
        updated beta according to
        beta_{t + 1} = beta_t + 72gamma/ T sum_{i in S_t} e_i
    '''

    beta += 72 * gamma / T * sum([np.eye(len(beta))[i] for i in cut])
    return beta


# 5.7 FindCut subroutine
## ProjRound


def projection_rounding(graph: nx.Graph, embeddings: np.ndarray,
                        b: float) -> set:
    '''
    Input:
        embeddings: a list of n-dimensional vectors, where each vector is an embedding of the node v_i
        b: a constant balance value b \in (0, 1/2]
    Output:
        cut: a subset of V
    '''
    n = len(embeddings)
    m = graph.number_of_edges()
    h = len(embeddings[0])
    T = int(np.ceil(np.log(n)))
    c = b / 100  # constant c = Omega(b) <= b/100
    for t in range(1, T + 1):
        # pick a unit vector u uniformly at random from S^h-1
        u = random_unit_vectors(k=h, n=1)

        # define x_i = sqrt{h} u^T v_i
        x = np.sqrt(h) * np.dot(u.T, embeddings)

        _, _, cut = sweep_cut(graph, x, lower=c * 2 * m, upper=(1 - c) * 2 * m)
    return cut


## FindCut
def find_cut(graph: nx.Graph, b: float, alpha: float, embeddings: np.ndarray,
             gamma: float) -> np.ndarray | None:
    '''
    Input:
        graph: an unweighted instance graph G = (V,E)
        b: a constant balance value b \in (0, 1/2]
        alpha: a constant alpha > 0
        embeddings: a list of n-dimensional vectors, where each vector is an embedding of the node v_i
    Output:
        cut: a subset of V
        is_balanced: True if the cut is balanced, False otherwise
    '''
    v_avg = compute_average_node_embedding(graph, embeddings)
    r = np.linalg.norm(embeddings - v_avg, axis=1)

    # compute the gram matrix X of embeddings
    X = embeddings.T @ embeddings

    L_K_V = compute_laplacian_complete_graph(graph)
    L = nx.laplacian_matrix(graph, weight="weight")
    psi = np.trace(L_K_V @ X)

    # the set R := {i in V : r_i^2 <= 32 * (1-b)/b * psi / 2m}
    graph_R = graph.copy()
    for i in graph.nodes():
        if r[i]**2 > 32 * (1 - b) / b * psi / (2 * m):
            graph_R.remove_node(i)

    L_K_R = compute_laplacian_complete_graph(graph_R)

    if np.trace(L @ X) > alpha * psi:
        return None

    if np.trace(L_K_R @ X) >= alpha * psi / 128:
        cut = projection_rounding(graph_R, embeddings, b)
        return cut

    # run sweep cut on G with embedding r. Let z be the smallest index such that vol(S_z) >= b/4 * 2m
    # output the most balanced sweep cut among {S1, ..., Sz-1} such that the conductance is at most 40 \sqrt{gamma}
    # if no such cut exists, raise an error

    _, conductance, cut = sweep_cut(graph, r, partial=b / 4 * 2 * m)
    if conductance <= 40 * np.sqrt(gamma):
        return cut
    else:
        raise ValueError("No balanced cut exists")


# Final algorithm
def balanced_separator(graph, b, gamma, epsilon=1, alpha=1) -> set | None:
    '''
    input:
        graph: an unweighted instance graph G = (V,E)
        b: a constant balance value b \in (0, 1/2]
        gamma: conductance value gamma in [1/n^2, 1)
    Output:
        cut: a balanced cut of G that has conductance at most gamma and balance at least b
        None if no such cut exists
    '''

    n = len(graph)
    T = 12 * int(np.ceil(np.log(n)))
    S = set()
    beta = np.zeros(n)  # Initialize beta

    # Run for T iterations
    # Iteration 1, beta = 0, P is the probability transition matrix of the heat kernel on G for time Tau
    #

    for t in range(1, T + 1):
        # compute heat kernel
        L = nx.laplacian_matrix(graph, weight="weight")
        degree_vector = nx.degree_matrix(graph, weight='weight')
        D_sqrt_inv = np.diag(1 / np.sqrt(degree_vector))
        P_t = compute_transition_matrix(graph, L, beta, degree_vector, t)

        # Generate random unit vectors
        k = int(np.ceil(np.log(n) / alpha))
        unit_vectors = random_unit_vectors(k=k, n=n)

        # compute the embedding vectors given by the expression
        #   (v_i^t)_j = \sqrt{n/k} u_j^T D^-1/2 P^t e_i
        vectors = np.zeros((n, k))
        vectors = np.sqrt(n / k) * np.dot(unit_vectors.T, D_sqrt_inv) @ P_t
        # TODO: replace with EXPV
        # vectors = EXPV(P_t, unit_vectors)

        # construct X_t, the gram matrix coreesponding to the vectors
        X_t = vectors.T @ vectors

        # check if the approximation to the total deviation is within (1 + epsilon)/n

        # compute the special complete graph laplacian
        L_K_V = compute_laplacian_complete_graph(graph)
        # compute  L_K_V \bullet D^-1 X_t
        estimate = np.trace(L_K_V @ X_t)
        if estimate > (1 + epsilon) / n:
            return None

        # Run FindCut subroutine
        cut = find_cut(graph, b, gamma, vectors)
        if not cut:
            return None

        # Check if the cut is balanced
        is_balanced = check_balanced(graph, cut, b)
        if is_balanced:
            return cut  # Return the balanced cut
        else:
            S.update(cut)  # Update set S with the new cut

        # Update beta for the next iteration
        beta = update_beta(beta, cut, t, T, S)

    return None  # If no balanced cut is found, return False
