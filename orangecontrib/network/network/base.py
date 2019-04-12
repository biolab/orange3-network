from functools import reduce, wraps, partial
from typing import Sequence

import numpy as np
import scipy.sparse as sp


class Edges:
    def __init__(self,
                 edges: sp.csr_matrix,  # row=from, column=to
                 edge_data: Sequence = None,
                 directed: bool = False,
                 name: str = ""):
        # for undirected graphs, does out_edges contains both directions?!
        # currently doesn't. If it however would, it would hurt nxexplorer!!!
        self.out_edges = edges.tocsr()
        self.in_edges = edges.transpose().tocsr()
        self.edge_data = edge_data
        self.directed = directed
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
        edges = self.edges[edge_type]
        if edges.directed:
            return edges.in_edges.shape[0] + edges.out_edges.shape[0]
        else:
            return edges.out_edges.shape[0] // 2

    def links(self, attr, edge_type=0, matrix_type=sp.coo_matrix):
        edges = self.edges[edge_type]
        return matrix_type(edges.edges), edges.edge_data.get_column_view(attr)

    def weighted_links(self, edge_type=0, matrix_type=sp.coo_matrix):
        return matrix_type(self.edges[edge_type].edges)

    @sum_over_edge_types()
    def out_degrees(self, edge_type):
        out_edges = self.edges[edge_type].out_edges
        return out_edges.indptr[1:] - out_edges.indptr[:-1]

    @sum_over_edge_types()
    def in_degrees(self, edge_type=None):
        if self.edges[edge_type].directed:
            in_edges = self.edges[edge_type].in_edges
            return in_edges.indptr[1:] - in_edges.indptr[:-1]
        else:
            return self.out_degrees(edge_type)

    @sum_over_edge_types()
    def degrees(self, edge_type=None):
        if self.edges[edge_type].directed:
            return self.out_degrees(edge_type) + self.in_degrees(edge_type)
        else:
            return self.out_degrees(edge_type)

    @sum_over_edge_types(1)
    def out_degree(self, node, edge_type=None):
        out_edges = self.edges[edge_type].out_edges
        return out_edges.indptr[node + 1] - out_edges.indptr[node]

    @sum_over_edge_types(1)
    def in_degree(self, node, edge_type=None):
        if not self.edges[edge_type].directed:
            return self.out_degree(edge_type)
        in_edges = self.edges[edge_type].in_edges
        return in_edges.indptr[node + 1] - in_edges.indptr[node]

    @sum_over_edge_types(1)
    def degree(self, node, edge_type=None):
        if self.edges[edge_type].directed:
            return self.in_degree(node, edge_type) \
                   + self.out_degree(node, edge_type)
        else:
            return self.out_degree(node, edge_type)

    @staticmethod
    def _compose_neighbours(node, matrix, weights):
        fr, to = matrix.indptr[node], matrix.indptr[node + 1]
        if not weights:
            return matrix.indices[fr:to]
        else:
            return np.vstack(np.atleast_2d(matrix.indices[fr:to]),
                             np.atleast_2d(matrix.data[fr:to]))

    @concatenate_over_edge_types(1)
    def outgoing(self, node, edge_type=None, weights=False):
        matrix = self.edges[edge_type].out_edges
        return self._compose_neighbours(node, matrix, weights)

    @concatenate_over_edge_types(1)
    def ingoing(self, node, edge_type=None, weights=False):
        edges = self.edges[edge_type]
        matrix = edges.in_edges if edges.directed else edges.out_edges
        return self._compose_neighbours(node, matrix, weights)

    @concatenate_over_edge_types(1)
    def neighbours(self, node, edge_type=None, weights=False):
        if not self.edges[edge_type].directed:
            return self.outgoing(node, edge_type, weights)
        else:
            return np.hstack((self.outgoing(node, edge_type, weights),
                              self.ingoing(node, edge_type, weights)))

    def subgraph(self, mask):
        def subset_edges(matrix, edge_data):
            if matrix is None:
                return None, None
            matrix = matrix.tocoo()
            edge_mask = np.logical_and(mask[matrix.row], mask[matrix.col])
            row = node_renumeration[matrix.row[edge_mask]]
            col = node_renumeration[matrix.col[edge_mask]]
            data = matrix.data[edge_mask]
            edge_data = edge_data[edge_mask] if edge_data is not None else None
            return sp.csr_matrix((data, (row, col)), shape=shape), edge_data

        nodes = self.nodes[mask]
        coordinates = self.coordinates[mask]
        if mask.dtype is not np.bool:
            mask1 = np.full((self.number_of_nodes(),), False)
            mask1[mask] = True
            mask = mask1
        node_renumeration = np.cumsum(mask) - 1
        shape = (len(nodes), len(nodes))
        edge_sets = [Edges(*subset_edges(edges.out_edges, edges.edge_data),
                           edges.directed, edges.name)
                     for edges in self.edges]
        return Network(nodes, edge_sets, self.name, coordinates)
