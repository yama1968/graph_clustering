#

from scipy.sparse import csr_matrix, diags, eye
import numpy as np


def vertex_degree(matrix, vertex):
    """Compute the degree of vertex in graph defined by sparse (symmetric) matrix."""

    return matrix[:, vertex].sum(axis=0)[0, 0]


def group_vertex_degree(matrix, group):
    """Compute the vertex degree of a group:
the number of vertices outside of the group to which it is connected."""

    return (matrix[~ group, :][:, group].sum(axis=1) != 0).sum()


def group_edge_degree(matrix, group):
    """Compute the edge degree of a group:
the number of edges going from the group to the outside."""

    return (matrix[~ group, :][:, group].sum(axis=1)).sum()


def couples_to_sparse_matrix(l, n_vertices):
    """Compute the sparse matrix of a graph from the list of edges and number of vertices"""

    m = csr_matrix((np.ones(len(l)), tuple(zip(*l))),
                   shape=(n_vertices, n_vertices))
    return m + m.T


def s2f(m, n):

    m = set(m)

    return np.array([x in m for x in range(n)])
