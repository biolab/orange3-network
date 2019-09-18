import shlex

import numpy as np
import scipy.sparse as sp

from .base import Network, EdgeType

__all__ = ("read_pajek", )


def read_vertices(lines):
    coordinates = np.zeros((len(lines), 2))
    labels = np.full((len(lines)), None, dtype=object)
    id_idx = {}
    for i, line in enumerate(lines):
        node_id, *parts = shlex.split(line)[:4]
        if not parts:
            label = str(node_id)
        else:
            try:  # The format specification was never set in stone, it seems
                label, x, y, *_ = parts + [None, None]
                x, y = float(x), float(y)
                coordinates[i, :2] = (x, y)
            except (TypeError, ValueError):
                if x is not None:
                    label = line.strip().split(maxsplit=1)[1]
        id_idx[node_id] = i
        labels[i] = label
    if np.sum(np.abs(coordinates)) == 0:
        coordinates = None
    return id_idx, labels, coordinates


def read_edges(id_idx, lines, nvertices):
    lines = [(id_idx[v1], id_idx[v2], abs(float(value)))
             for v1, v2, value, *_ in (line.split()[:3] + [1]
                                       for line in lines)]
    v1s, v2s, values = zip(*lines)
    values = np.array(values)
    if values.size and np.all(values == values[0]):
        values = np.lib.stride_tricks.as_strided(
            values[0], (len(values), ), (0, ))
        values.flags.writeable = False
    return sp.coo_matrix((values, (np.array(v1s), np.array(v2s))),
                         shape=(nvertices, nvertices))


def read_edges_list(id_idx, lines, nvertices):
    indptr = []
    indices = []
    lines = sorted((id_idx[source], [id_idx[t] for t in targets])
                   for source, *targets in (line.split() for line in lines))
    # Todo: We can avoid using potentially long Python list by writing directly
    # into np.arrays
    for source, targets in lines:
        indptr += [len(indices)] * (source - len(indptr))
        indices += targets
    indptr += [len(indices)] * (nvertices - len(indptr) + 1)
    indices = np.array(indices, dtype=int)
    data = np.lib.stride_tricks.as_strided(np.ones(1), (len(indices), ), (0, ))
    data.flags.writeable = False
    return sp.csr_matrix((data, indices, indptr), shape=(nvertices, nvertices))


def read_pajek(path):
    def check_has_vertices():
        if labels is None:
            raise ValueError("Vertices must be defined before edges or arcs")

    with open(path) as f:
        lines = [line.strip() for line in f]
    lines = np.array([line for line in lines if line and line[0] != "%"])
    part_starts = [(line_no, line)
                   for line_no, line in enumerate(lines)
                   if line[0] == "*"]
    network_name = None
    edges = []
    labels = coordinates = None
    in_first_mode = None
    for (start_line, part_desc), (end_line, *_) \
            in zip(part_starts, part_starts[1:] + [(len(lines), "")]):
        part_type, part_args, *_ = part_desc.split(maxsplit=1) + [""]
        part_type = part_type.lower()
        line_part = lines[start_line + 1: end_line]
        if not line_part.size:
            continue
        if part_type == "*network":
            if network_name is not None:
                raise ValueError(
                    "Pajek files with multiple networks are not supported")
            network_name = part_args.strip()
        elif part_type == "*vertices":
            if labels is not None:
                raise ValueError(
                    "Pajek files with multiple set of vertices are not "
                    "supported")
            id_idx, labels, coordinates = read_vertices(line_part)
            part_args = part_args.split()
            if len(part_args) > 1:
                in_first_mode = int(part_args[1])
        elif part_type in ("*edges", "*arcs"):
            check_has_vertices()
            edges.append(
                EdgeType[part_type=="*arcs"](
                    read_edges(id_idx, line_part, len(labels)),
                    name=part_args.strip() or part_type[1:]))
        elif part_type in ("*edgeslist", "*arcslist"):
            check_has_vertices()
            edges.append(
                EdgeType[part_type=="*arcslist"](
                    read_edges_list(id_idx, line_part, len(labels)),
                    name=part_args.strip() or part_type[1:]))
    network = Network(labels, edges, network_name or "", coordinates)
    if in_first_mode is not None:
        network.in_first_mode = in_first_mode
    return network


# TODO: doesn't belong here if this module is meant as independent from Orange
def transform_data_to_orange_table(network: Network):
    from Orange.data import (
        Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable)
    n_nodes = network.number_of_nodes()
    attrs = []
    if network.coordinates is not None:
        attrs += [ContinuousVariable('x'), ContinuousVariable('y')]
        coords = network.coordinates
    else:
        coords = np.zeros((n_nodes, 0), dtype=float)
    if getattr(network, "in_first_mode", False):
        attrs.append(DiscreteVariable("mode", values=("0", "1")))
        modes = np.ones((n_nodes, 1), dtype=float)
        modes[:network.in_first_mode] = 0
    else:
        modes = np.zeros((n_nodes, 0), dtype=float)
    domain = Domain(attrs, metas=[StringVariable('label ')])
    network.nodes = Table.from_numpy(
        domain,
        np.hstack((coords, modes)),
        metas=network.nodes.reshape(n_nodes, 1))
