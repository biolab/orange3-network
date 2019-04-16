from functools import reduce, wraps, partial
from typing import Sequence

import numpy as np
import scipy.sparse as sp


class Edges:
    directed = None

    def __init__(self,
                 edges,  # row=from, column=to
                 edge_data: Sequence = None,
                 name: str = ""):
        self.edges = edges.tocsr(copy=True)
        self.edges.sum_duplicates()
        self.edge_data = edge_data
        self.name = name

    def out_degrees(self) -> np.ndarray:
        pass

    def in_degrees(self) -> np.ndarray:
        pass

    def degrees(self) -> np.ndarray:
        pass

    def out_degree(self, node) -> float:
        pass

    def in_degree(self, node) -> float:
        pass

    def degree(self, node) -> float:
        pass

    def outgoing(self, node, weights=False) -> np.ndarray:
        pass

    def ingoing(self, node, weights=False) -> np.ndarray:
        pass

    def neighbours(self, node, edge_type=None, weights=False) -> np.ndarray:
        pass

    @staticmethod
    def _compose_neighbours(node: int, matrix: sp.csr_matrix, weights: bool):
        fr, to = matrix.indptr[node], matrix.indptr[node + 1]
        if not weights:
            return matrix.indices[fr:to]
        else:
            return np.vstack(np.atleast_2d(matrix.indices[fr:to]),
                             np.atleast_2d(matrix.data[fr:to]))

    def subset(self, mask, node_renumeration, shape):
        edges = self.edges.tocoo()
        edge_mask = np.logical_and(mask[edges.row], mask[edges.col])
        row = node_renumeration[edges.row[edge_mask]]
        col = node_renumeration[edges.col[edge_mask]]
        data = self.data[edge_mask]
        edge_data = self.edge_data[edge_mask] if self.edge_data is not None \
            else None
        return type(self)(
            sp.csr_matrix(
                (data, (row, col)), shape=shape), edge_data, self.name)


class DirectedEdges(Edges):
    directed = True

    def __init__(self,
                 edges: sp.csr_matrix,  # row=from, column=to
                 edge_data: Sequence = None,
                 name: str = ""):
        super().__init__(edges, edge_data, name)
        self.in_edges = self.edges.transpose()

    def out_degrees(self):
        return self.edges.indptr[1:] - self.edges.indptr[:-1]

    def in_degrees(self):
        return self.in_edges.indptr[1:] - self.in_edges.indptr[:-1]

    def degrees(self):
        return self.out_degrees() + self.in_degrees()

    def out_degree(self, node):
        return self.edges.indptr[node + 1] - self.edges.indptr[node]

    def in_degree(self, node):
        return self.in_edges.indptr[node + 1] - self.in_edges.indptr[node]

    def degree(self, node):
        return self.out_degree(node) + self.in_degree(node)

    def outgoing(self, node, weights=False):
        return self._compose_neighbours(node, self.edges, weights)

    def incoming(self, node, weights=False):
        return self._compose_neighbours(node, self.in_edges, weights)

    def neighbours(self, node, edge_type=None, weights=False):
        return np.hstack((self.outgoing(node, weights),
                          self.ingoing(node, weights)))


class UndirectedEdges(Edges):
    directed = False

    def __init__(self,
                 edges: sp.csr_matrix,
                 edge_data: Sequence = None,
                 name: str = ""):
        super().__init__(edges, edge_data, name)
        self.twoway_edges = self.edges + self.edges.transpose()
        self.twoway_edges.sum_duplicates()

    def degrees(self):
        return self.twoway_edges.indptr[1:] - self.twoway_edges.indptr[:-1]

    def degree(self, node):
        return self.edges.indptr[node + 1] - self.edges.indptr[node]

    def neighbours(self, node, weights=False):
        return self._compose_neighbours(node, self.twoway_edges, weights)

    in_degrees = out_degrees = degrees
    in_degree = out_degree = degree
    incoming = outgoing = neighbours


EdgeType = [UndirectedEdges, DirectedEdges]


def aggregate_over_edge_types(aggregate, arg_no=0):
    def wrapwrap(f):
        @wraps(f)
        def wrapper(graph, *args):
            if len(args) <= arg_no or args[arg_no] is None:
                return aggregate(
                    f(graph, *args[:arg_no], edge_type, *args[arg_no + 1:])
                    for edge_type in range(len(graph.edges)))
            else:
                return f(graph, *args)
        return wrapper
    return wrapwrap


sum_over_edge_types = \
    partial(aggregate_over_edge_types, lambda x: reduce(np.add, x))

concatenate_over_edge_types = partial(aggregate_over_edge_types, np.hstack)


class Network:
    def __init__(self, nodes: Sequence, edges: Sequence, name: str = "",
                 coordinates: np.ndarray = None):
        """
        Attributes:
            nodes (Sequence): data about nodes; it can also be just range(n)
            edges (List[Edges]): one or more set of edges
            name (str): network name

        Args:
            nodes (Sequence): data about nodes; it can also be just range(n)
            edges (sp.spmatrix or Edges or List[Edges]):
                one or more set of edges
            name (str): network name
        """
        self.nodes = nodes
        if sp.issparse(edges):
            edges = UndirectedEdges(edges)
        if isinstance(edges, Edges):
            edges = [edges]
        self.edges = edges
        self.name = name
        self.coordinates = coordinates

    def number_of_nodes(self):
        return len(self.nodes)

    @sum_over_edge_types()
    def number_of_edges(self, edge_type=None):
        return len(self.edges[edge_type].edges.indices)

    def links(self, attr, edge_type=0, matrix_type=sp.coo_matrix):
        edges = self.edges[edge_type]
        return matrix_type(edges.edges), edges.edge_data.get_column_view(attr)

    @sum_over_edge_types()
    def out_degrees(self, edge_type):
        return self.edges[edge_type].out_degrees()

    @sum_over_edge_types()
    def in_degrees(self, edge_type=None):
        return self.edges[edge_type].in_degrees()

    @sum_over_edge_types()
    def degrees(self, edge_type=None):
        return self.edges[edge_type].degrees()

    @sum_over_edge_types(1)
    def out_degree(self, node, edge_type=None):
        return self.edges[edge_type].out_degree(node)

    @sum_over_edge_types(1)
    def in_degree(self, node, edge_type=None):
        return self.edges[edge_type].in_degree(node)

    @sum_over_edge_types(1)
    def degree(self, node, edge_type=None):
        return self.edges[edge_type].degree(node)

    @concatenate_over_edge_types(1)
    def outgoing(self, node, edge_type=None, weights=False):
        return self.edges[edge_type].outgoing(node, weights)

    @concatenate_over_edge_types(1)
    def incoming(self, node, edge_type=None, weights=False):
        return self.edges[edge_type].incoming(node, weights)

    @concatenate_over_edge_types(1)
    def neighbours(self, node, edge_type=None, weights=False):
        return self.edges[edge_type].neighbours(node, weights)

    def subgraph(self, mask):
        nodes = self.nodes[mask]
        coordinates = self.coordinates[mask]
        if mask.dtype is not np.bool:
            mask1 = np.full((self.number_of_nodes(),), False)
            mask1[mask] = True
            mask = mask1
        node_renumeration = np.cumsum(mask) - 1
        shape = (len(nodes), len(nodes))
        edge_sets = [edges.subset(mask, node_renumeration, shape)
                     for edges in self.edges]
        return Network(nodes, edge_sets, self.name, coordinates)