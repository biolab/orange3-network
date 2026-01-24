from typing import Union, Dict, Optional

import numpy as np
import scipy.sparse as sp

from Orange.data import Table, StringVariable, ContinuousVariable, Domain
from orangecontrib.network import Network
from orangecontrib.network.network.base import DirectedEdges, UndirectedEdges

MAX_LABELS = 100_000


class ComposeError(Exception):
    pass


class NonUniqueLabels(ComposeError):
    pass


class MismatchingEdgeVariables(ComposeError):
    pass


class UnknownNodes(ComposeError):
    pass


def network_from_tables(
        data: Table,
        label_variable: StringVariable,
        edges: Table,
        edge_src_variable: Union[StringVariable, ContinuousVariable],
        edge_dst_variable: Union[StringVariable, ContinuousVariable],
        directed=False) -> Network:

    labels = data.get_column(label_variable)
    label_idcs = {label: i for i, label in enumerate(labels)}
    if len(label_idcs) < len(labels):
        raise NonUniqueLabels()

    src_col, dst_col = _edge_columns(edges, edge_src_variable, edge_dst_variable)
    if isinstance(edge_src_variable, ContinuousVariable):
        row_ind = _float_to_ind(src_col, edge_src_variable.name, len(data))
        col_ind = _float_to_ind(dst_col, edge_dst_variable.name, len(data))
    else:
        row_ind = _str_to_ind(src_col, label_idcs)
        col_ind = _str_to_ind(dst_col, label_idcs)

    edge_data = _reduced_edge_data(edges, edge_src_variable, edge_dst_variable)
    return _net_from_data_and_edges(data, edge_data, row_ind, col_ind, directed)


def network_from_edge_table(
        edges: Table,
        edge_src_variable: Union[StringVariable, ContinuousVariable],
        edge_dst_variable: Union[StringVariable, ContinuousVariable],
        directed=False) -> Network:

    src_col, dst_col = _edge_columns(edges, edge_src_variable, edge_dst_variable)
    if isinstance(edge_src_variable, ContinuousVariable):
        row_ind = _float_to_ind(src_col, edge_src_variable.name)
        col_ind = _float_to_ind(dst_col, edge_dst_variable.name)
        labels = [str(x)
                  for x in range(1, max(np.max(row_ind), np.max(col_ind)) + 2)]
    else:
        labels = sorted(set(src_col) | set(dst_col))
        label_idcs = {label: i for i, label in enumerate(labels)}
        row_ind = _str_to_ind(src_col, label_idcs)
        col_ind = _str_to_ind(dst_col, label_idcs)

    domain = Domain([], [], [StringVariable("node_label")])
    n = len(labels)
    labels = Table.from_numpy(
        domain, np.empty((n, 0)), np.empty((n, 0)), np.array([labels]).T)

    edge_data = _reduced_edge_data(edges, edge_src_variable, edge_dst_variable)
    return _net_from_data_and_edges(labels, edge_data, row_ind, col_ind, directed)


def _net_from_data_and_edges(data, edge_data, row_ind, col_ind, directed=False):
    assert len(row_ind) == len(col_ind)

    if edge_data is not None:
        assert len(row_ind) == len(edge_data)
        edge_data = _sort_edges(row_ind, col_ind, edge_data)

    ones = np.lib.stride_tricks.as_strided(np.ones(1), (len(row_ind),), (0,))
    edge_type = DirectedEdges if directed else UndirectedEdges
    net_edges = edge_type(
        sp.csr_array((ones, (row_ind, col_ind)), shape=(len(data), ) * 2),
        edge_data)
    return Network(data, net_edges)


def _sort_edges(row_ind, col_ind, edge_data):
    indices = np.lexsort((col_ind, row_ind))
    return edge_data[indices]


def _reduced_edge_data(edges, edge_src_variable, edge_dst_variable):
    domain = edges.domain
    parts = [[var for var in part
              if var not in (edge_src_variable, edge_dst_variable)]
             for part in (domain.attributes, domain.class_vars, domain.metas)]
    if not any(parts):
        return None
    return edges.transform(Domain(*parts))


def _edge_columns(edges, edge_src_variable, edge_dst_variable):
    if type(edge_src_variable) is not type(edge_dst_variable):
        raise MismatchingEdgeVariables()

    return (edges.get_column(edge_src_variable),
            edges.get_column(edge_dst_variable))


def _str_to_ind(col: np.ndarray, label_idcs: Dict[str, int]) -> np.ndarray:
    ind = np.fromiter((label_idcs.get(x, -1) for x in col),
                      count=len(col), dtype=int)
    if np.min(ind) == -1:
        raise UnknownNodes("Unknown labels: "
                           + ", ".join(sorted(set(col) - set(label_idcs))))
    return ind


def _float_to_ind(col: np.ndarray,
                  var_name: str,
                  nlabels: Optional[int] = None) -> np.ndarray:
    mi, ma = np.min(col), np.max(col)
    if mi < 0:
        raise UnknownNodes("negative vertex indices")
    elif mi == 0:
        raise UnknownNodes("vertex indices must be 1-based")
    elif ma > (nlabels or MAX_LABELS):
        raise UnknownNodes("some indices are too large")
    elif np.isnan(mi) or np.isnan(ma):
        raise UnknownNodes(f"{var_name} has missing values")
    elif not np.all(np.modf(col)[0] == 0):
        raise UnknownNodes("some indices are non-integer")
    return col.astype(int) - 1
