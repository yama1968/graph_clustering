#

from gmet import *
import numpy as np


tests = {
    "eight": (9, [(0, 1), (0, 2), (1, 2), (0, 4), (4, 5), (4, 6), (5, 6), (4, 7)]),
    "double": (11, [(0, 1), (0, 2), (1, 2), (0, 4),
                    (4, 5), (4, 6), (5, 6),
                    (0, 7), (7, 8), (8, 9), (8, 10)])
}


def test_couples_to_sparse_matrix():

    for t in tests:

        x = tests[t]
        m = couples_to_sparse_matrix(x[1], x[0])

        assert m.shape == (x[0], x[0])
        assert (m.data != 1).sum() == 0

        nz = m.nonzero()

        assert m.sum() == 2 * len(x[1])


def test_eight_degree():

    x = tests["eight"]
    m = couples_to_sparse_matrix(x[1], x[0])

    assert vertex_degree(m, 4) == 4
    assert vertex_degree(m, 7) == 1
    assert vertex_degree(m, 8) == 0


def test_eight_vertex_degree():

    x = tests["eight"]
    m = couples_to_sparse_matrix(x[1], x[0])
    g1 = set([4, 5, 6])
    g1 = np.array([x in g1 for x in range(x[0])])

    assert len(g1) == m.shape[0]
    assert group_vertex_degree(m, g1) == 2
    assert group_vertex_degree(m, s2f([8], x[0])) == 0
    assert group_vertex_degree(m, s2f([5, 6], x[0])) == 1
    assert group_vertex_degree(m, s2f([7], x[0])) == 1

    assert group_vertex_degree(m, np.repeat(True, x[0])) == 0
    assert group_vertex_degree(m, np.repeat(False, x[0])) == 0


def test_eight_edge_degree():

    x = tests["eight"]
    m = couples_to_sparse_matrix(x[1], x[0])
    g1 = set([4, 5, 6])
    g1 = np.array([x in g1 for x in range(x[0])])

    assert group_edge_degree(m, g1) == 2
    assert group_edge_degree(m, s2f([8], x[0])) == 0
    assert group_edge_degree(m, s2f([5, 6], x[0])) == 2
    assert group_edge_degree(m, s2f([7], x[0])) == 1

    assert group_edge_degree(m, np.repeat(True, x[0])) == 0
    assert group_edge_degree(m, np.repeat(False, x[0])) == 0


def test_subg_size():

    x = tests["eight"]
    m = couples_to_sparse_matrix(x[1], x[0])
    g1 = set([4, 5, 6])
    g1 = np.array([x in g1 for x in range(x[0])])

    assert subg_size(m, g1) == 3
    assert subg_size(m, s2f({}, x[0])) == 0


def test_subg_vol():

    x = tests["eight"]
    m = couples_to_sparse_matrix(x[1], x[0])
    g1 = set([4, 5, 6])
    g1 = np.array([x in g1 for x in range(x[0])])

    print(subg_vol(m, g1), subg_vol(m, s2f([4], x[0])))

    assert subg_vol(m, g1) == 8
    assert subg_vol(m, s2f([4], x[0])) == 4
    assert subg_vol(m, s2f([5, 6], x[0])) == 4
    assert subg_vol(m, s2f({}, x[0])) == 0
