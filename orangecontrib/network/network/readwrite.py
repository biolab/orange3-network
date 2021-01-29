import shlex
from itertools import count, repeat

import numpy as np
import scipy.sparse as sp

from Orange.data.io import FileFormatMeta
from .base import Network, EdgeType

__all__ = ("read_pajek", "write_pajek", "PajekReader")


class NetFileFormat(metaclass=FileFormatMeta):
    pass


class PajekReader(NetFileFormat):
    EXTENSIONS = ('.net', )
    DESCRIPTION = 'Pajek network file format'

    @staticmethod
    def read(filename):
        return read_pajek(filename)

    @staticmethod
    def write(filename, network, labels=None):
        return write_pajek(filename, network, labels)


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
    lines = {id_idx[source]: [id_idx[t] for t in targets]
             for source, *targets in (line.split() for line in lines)}
    # Todo: We can avoid using potentially long Python list by writing directly
    # into np.arrays
    indptr = np.zeros(nvertices + 1, dtype=int)
    indices = []
    for idx in range(nvertices):
        targets = lines.get(idx, ())
        indices += targets
        indptr[idx + 1] = len(indices)
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


def write_pajek(path, network, labels=None):
    if len(network.edges) > 1:
        raise TypeError(
            "This implementation of Pajek format does not support saving "
            "networks with multiple edge types.")
    f = open(path, "wt") if isinstance(path, str) else path
    f.write(f'*Network "{network.name}"\n')
    _write_vertices(f, network, labels)
    if network.edges:
        _write_edges(f, network)
    if f is not path:
        f.close()


def _write_vertices(f, network, labels=None):
    if labels is None:
        labels = network.nodes

    f.write(f"*Vertices\t{network.number_of_nodes()}\n")
    if network.coordinates is not None:
        coords = network.coordinates
    else:
        coords = repeat(())
    for i, label, coordinates in zip(count(start=1), labels, coords):
        f.write(f'{i:6} "{label}"\t' +
                f"{' '.join(f'{c:.4f}' for c in coordinates)}\n")


def _write_edges(f, network):
    edges = network.edges[0]
    f.write("*Arcs\n" if edges.directed else "*Edges\n")
    mat = edges.edges
    if np.all(mat.data == 1):
        for row, rb, re in zip(count(), mat.indptr, mat.indptr[1:]):
            f.write("".join(f"{row + 1:6} {col + 1:6}\n"
                            for col in mat.indices[rb:re]))
    else:
        for row, rb, re in zip(count(), mat.indptr, mat.indptr[1:]):
            f.write("".join(f"{row + 1:6} {mat.indices[i] + 1:6} "
                            f"{mat.data[i]:.6f}\n" for i in range(rb, re)))


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
