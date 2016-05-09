#

import networkx as nx
import numpy as np


def compute_sub_adj(sparse_m,
                    cond):
    return (sparse_m[:, cond] != 0).sum(axis=1).nonzero()[0]


renderer = {
    "std":        nx.draw_networkx,
    "spectral":   nx.draw_spectral,
    "spring":     nx.draw_spring
}


def get_deg_sub(sparse_m,
                cond):
    out_links = sparse_m[cond, :].sum(axis=0)

    return ((out_links != 0) & ~ cond).sum()


def plot_subgraph_links(sparse_m, query, degree=0, layout="std", graph=None):

    cond = np.where(query)[0]

    if graph is None:
        graph = nx.from_scipy_sparse_matrix(sparse_m)

    if degree == 0:
        sub1 = cond
        node_color = "r"
    elif degree == 1:
        sub1 = list(set(cond) | set(
            compute_sub_adj(sparse_m, cond)))
 #       print(sub1)
        node_color = [("r" if (n in cond) else "b") for n in sub1]
 #       print(node_color)
    elif degree == 2:
        sub0 = set(cond) | set(compute_sub_adj(sparse_m, cond))
        sub1 = list(sub0 | set(compute_sub_adj(sparse_m, list(sub0))))
        node_color = [("r" if (n in cond) else "b" if (
            n in sub0) else "y") for n in sub1]

    renderer[layout](
        graph.subgraph(sub1),
        nodelist=list(sub1),
        node_color=node_color,
        alpha=0.5,
        labels={n: str(n) for n in sub1})
