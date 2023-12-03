import numpy as np
import networkx as nx
import scipy
from functools import lru_cache

from sweep_cut import sweep_cut


def random_unit_vectors(k: int, n: int) -> np.ndarray:
    # Generate k random vectors in R^n
    vectors = np.random.randn(k, n)
    # Normalize each vector to have unit length
    unit_vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    return unit_vectors


def update_beta(beta, cut, gamma, T) -> np.ndarray:
    '''
    Input:
        beta: a vector of length n
        cut: a subset of V in one hot
        t: current iteration
        T: total number of iterations
    Output:
        updated beta according to
        beta_{t + 1} = beta_t + 72gamma/ T sum_{i in S_t} e_i
    '''

    beta += 72 * gamma / T * sum([np.eye(len(beta))[i] for i in cut])
    return beta


def get_degree_vector(graph: nx.Graph) -> np.ndarray:
    return np.array(graph.degree(weight='weight'), dtype=np.float64)[:, 1]


## Fact 5.1
def compute_laplacian_complete_graph(graph: nx.Graph) -> np.ndarray:
    '''
    Input:
        graph: an unweighted instance graph G = (V,E)
    Output:
        laplacian of the special complete graph obtained via the function complete_graph()
    '''

    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    ones = np.ones((n, 1))

    D = np.diag(get_degree_vector(graph))
    L = D - 1 / (2 * m) * D @ ones @ ones.T @ D 
    return L


# Section 5.7 Projection rounding
def projection_rounding(graph, embeddings: np.ndarray, b: float) -> set:
    '''
    Input:
        embeddings: a list of n-dimensional vectors, where each vector is an embedding of the node v_i
        b: a constant balance value b \in (0, 1/2]
    Output:
        cut: a subset of V
    '''
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    h = len(embeddings[0])
    T = int(np.ceil(np.log(n)))
    c = b / 100  # constant c = Omega(b) <= b/100
    conductance_lst = []
    cut_lst = []
    for t in range(1, T + 1):
        # pick a unit vector u uniformly at random from S^h-1
        u = random_unit_vectors(k=h, n=1)

        # define x_i = sqrt{h} u^T v_i
        x = np.sqrt(h) * np.dot(u.T, embeddings)

        _, conductance, cut = sweep_cut(graph,
                                        x,
                                        lower_vol=c * 2 * m,
                                        upper_vol=(1 - c) * 2 * m)
        conductance_lst.append(conductance)
        cut_lst.append(cut)
    # return the cut with the smallest conductance
    idx = np.argmin(conductance_lst)
    return cut_lst[idx]


class BalSep:

    def __init__(self, graph) -> None:
        self.graph = graph

        self.degree_vector = get_degree_vector(self.graph)
        self.degree_matrix = np.diag(self.degree_vector)

        self.volume_graph = sum(self.degree_vector) 
        self.n = self.graph_nodes()
        self.m = self.graph_edges()
        self.laplacian_matrix = np.array(nx.laplacian_matrix(graph,
                                                    weight="weight").toarray(), dtype=np.float64)

    # Code below defines some of the preliminary quantities and special matrices
    # section 5.1, basic preliminaries:
    ## instance graph and edge volume
    def graph_nodes(self) -> int:
        '''
        Input:
            graph: an unweighted instance graph G = (V,E)
        Output:
            The number of nodes in the graph
        '''
        return self.graph.number_of_nodes()

    def graph_edges(self) -> int:
        '''
        Input:
            graph: an unweighted instance graph G = (V,E)
        Output:
            The number of edges in the graph
        '''
        return self.graph.number_of_edges()

    def get_volume(self, cut: np.ndarray) -> int:
        '''
        Input:
            graph: an unweighted instance graph G = (V,E)
            cut: a np.ndarray of length n, where cut[i] = 1 if i is in the cut, 0 otherwise
        Output:
            The volume of the cut
        '''
        # for i in cut:
        #     volume += graph.degree(i, weight='weight')
        volume = np.dot(self.degree_vector, cut)
        return volume

    def check_balanced(self, cut: np.ndarray, b: float) -> bool:
        '''
        Input:
            graph: an unweighted instance graph G = (V,E)
            cut: a np.ndarray of length n, where cut[i] = 1 if i is in the cut, 0 otherwise
            b: a constant balance value b \in (0, 1/2]
        Output:
            True if the cut is balanced, False otherwise
        '''

        volume_cut = self.get_volume(cut)
        # find the complement of the cut
        cut_complement = 1 - cut
        volume_complement = self.get_volume(cut_complement)
        if min(volume_cut, volume_complement) >= b * self.volume_graph:
            return True
        return False

    ## speical graph
    def generate_complete_graph(self) -> nx.Graph:
        '''
        Input:
            graph: an unweighted instance graph G = (V,E)
        Output:
            A complete graph with weight d_i d_j / 2m between every pair i,j in V.
        '''

        complete_graph = nx.Graph()

        for i in self.n:
            for j in self.n:
                if i != j:
                    complete_graph.add_edge(i,
                                            j,
                                            weight=self.degree_vector[i] *
                                            self.degree_vector[j] /
                                            (2 * self.m))
        return complete_graph

    def generate_star_graph(self, v) -> nx.Graph:
        '''
        Generate a star graph based on the vertex v
        Input:
            graph: an unweighted instance graph G = (V,E)
            v: a vertex in V
        Output:
            S_i: the star grpah rooted at i, with edge weight of d_id_j/2m between i and j for all j in V
        '''
        star_graph = nx.Graph()
        # efficiently compute the sum of degree = 2m = 2 * {n choose 2} = 2 * n * (n - 1)/2
        for j in range(self.n):
            if j != v:
                star_graph.add_edge(v,
                                    j,
                                    weight=self.degree_vector[v] *
                                    self.degree_vector[j] / (2 * self.m))
        return star_graph

    ## Embedding Notation, v_avg
    def compute_average_node_embedding(self, embeddings) -> np.ndarray:
        '''
        Input:
            embeddings: a list of n-dimensional vectors, where each vector is an embedding of the node v_i
        Output:
            the mean vector of the embeddings defined as
            v_avg := sum_{i in V} d_i/(2m) * v_i
        '''
        v_avg = np.zeros(len(embeddings[0]))

        for (i, deg) in enumerate(self.degree_vector):
            v_avg += embeddings[i] * deg / (2 * self.m)

        return v_avg

    # Section 5.2, AHK random walk
    ## Denote P_tau (beta) as:
    ## Probability Transitimation Matrix of H(beta) between time 0 and tau
    ## P_tau (beta) = exp(tau Q(beta))
    ## where
    ## Q(beta) = -( L + sum_i in V beta_i L(S_i)) D^-1

    def compute_transition_rate_matrix(self, beta) -> np.ndarray:
        '''
        Input:
            L: the Laplacian matrix of the instance graph
            beta: a vector of length n
            degree_vector one dimension vector of the diagonal of the degree matrix of the instance graph
        Output:
            Q(beta) = -( L + sum_i in V beta_i L(S_i)) D^-1
        '''
        D_inv = np.diag(1 / self.degree_vector)
        Q = self.laplacian_matrix
        laplacian_star_graphs = [
            nx.laplacian_matrix(self.generate_star_graph(i),
                                weight="weight").toarray()
            for i in range(self.n)
        ]

        for i in range(self.n):
            Q += beta[i] * laplacian_star_graphs[i]
        Q = -Q @ D_inv
        return Q

    def compute_transition_matrix(self, beta, tau) -> np.ndarray:
        '''
        Input:
            L: the Laplacian matrix of the instance graph
            beta: a vector of length n
            degree_vector one dimension vector of the diagonal of the degree matrix of the instance graph
            tau: time parameter
        Output:
            P_tau (beta) = exp(tau Q(beta))
        '''
        Q = self.compute_transition_rate_matrix(beta)
        return scipy.linalg.expm(tau * Q)

    ## total deviation of a graph:
    ## psi(P_tau(beta), G) = L(K_V) \bullet D^-1 P_2tau(beta)
    def compute_total_deviation_graph(self, tau, beta) -> float:
        '''
        Input:
            P: a probability transition matrix
        Output:
            psi(P, G) = L(K_V) \bullet D^-1 P_2tau(beta)
        '''
        D_inv = np.diag(1 / self.degree_vector)
        P_2tau_beta = self.compute_transition_matrix(beta, 2 * tau)
        L_K_V = compute_laplacian_complete_graph(self.graph)

        return np.trace(L_K_V @ D_inv) @ P_2tau_beta

    # 5.7 FindCut subroutine
    ## ProjRound

    ## FindCut
    def find_cut(self, b: float, alpha: float, embeddings: np.ndarray,
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
        v_avg = self.compute_average_node_embedding(embeddings)
        r = np.linalg.norm(embeddings - v_avg, axis=1)

        # compute the gram matrix X of embeddings
        X = embeddings.T @ embeddings

        L_K_V = compute_laplacian_complete_graph(self.graph)
        psi = np.trace(L_K_V @ X)

        # the set R := {i in V : r_i^2 <= 32 * (1-b)/b * psi / 2m}
        graph_R = self.graph.copy()

        for i in self.n:
            if r[i]**2 > 32 * (1 - b) / b * psi / (2 * self.m):
                graph_R.remove_node(i)

        L_K_R = compute_laplacian_complete_graph(graph_R)

        if np.trace(self.laplacian_matrix @ X) > alpha * psi:
            return None

        if np.trace(L_K_R @ X) >= alpha * psi / 128:
            cut = projection_rounding(graph_R, embeddings, b)
            return cut

        # run sweep cut on G with embedding r. Let z be the smallest index such that vol(S_z) >= b/4 * 2m
        # output the most balanced sweep cut among {S1, ..., Sz-1} such that the conductance is at most 40 \sqrt{gamma}
        # if no such cut exists, raise an error

        _, conductance, cut = sweep_cut(self.graph,
                                        r,
                                        partial=b / 4 * 2 * self.m)
        if conductance <= 40 * np.sqrt(gamma):
            return cut
        else:
            raise ValueError("No balanced cut exists")

    # Final algorithm
    def balanced_separator(self, b, gamma, epsilon=1, alpha=1) -> set | None:
        '''
        input:
            graph: an unweighted instance graph G = (V,E)
            b: a constant balance value b \in (0, 1/2]
            gamma: conductance value gamma in [1/n^2, 1)
        Output:
            cut: a balanced cut of G that has conductance at most gamma and balance at least b
            None if no such cut exists
        '''

        T = 12 * int(np.ceil(np.log(self.n)))
        S = np.zeros(self.n)
        beta = np.zeros(self.n, dtype=np.float64)  # Initialize beta
        tau = np.log(self.n / 12 * gamma)

        # Run for T iterations
        # Iteration 1, beta = 0, P is the probability transition matrix of the heat kernel on G for time Tau
        #

        for t in range(1, T + 1):
            # compute heat kernel

            D_sqrt_inv = np.diag(1 / np.sqrt(self.degree_vector))
            P_t = self.compute_transition_matrix(beta, tau)

            # Generate random unit vectors
            k = int(np.ceil(np.log(self.n) / alpha))
            unit_vectors = random_unit_vectors(k=k, n=self.n)

            # compute the embedding vectors given by the expression
            #   (v_i^t)_j = \sqrt{n/k} u_j^T D^-1/2 P^t e_i
            vectors = np.zeros((self.n, k))

            vectors = np.sqrt(self.n / k) * unit_vectors @ D_sqrt_inv @ P_t
            # TODO: replace with EXPV
            # vectors = EXPV(P_t, unit_vectors)

            # construct X_t, the gram matrix coreesponding to the vectors
            X_t = vectors.T @ vectors

            # check if the approximation to the total deviation is within (1 + epsilon)/n

            # compute the special complete graph laplacian
            L_K_V = compute_laplacian_complete_graph(self.graph)
            # compute  L_K_V \bullet D^-1 X_t
            estimate = np.trace(L_K_V @ X_t)
            if estimate <= (1 + epsilon) / self.n:
                return None

            # Run FindCut subroutine
            cut = self.find_cut(b, alpha, vectors, gamma)
            if not cut:
                return None

            # Check if the cut is balanced
            is_balanced = self.check_balanced(cut, b)
            if is_balanced:
                return cut  # Return the balanced cut
            else:
                # S.update(cut)  # Update set S with the new cut
                S = np.logical_or(S, cut)

            # Update beta for the next iteration
            beta = update_beta(beta, cut, t, T)

        return None  # If no balanced cut is found, return False
