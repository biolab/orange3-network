import shlex
from collections import defaultdict
from itertools import count, repeat, chain
from math import isnan
from typing import Optional

import numpy as np
import scipy.sparse as sp

from Orange.data import ContinuousVariable, is_discrete_values, \
    DiscreteVariable, StringVariable, Domain, Table
from Orange.data.io import FileFormatMeta
from Orange.data.util import get_unique_names
from Orange.misc.collections import natural_sorted
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
    def write(filename,
              network,
              labels: list[str] = None,
              node_attributes: list[str] = None,
              edge_attributes: list[str] = None):
        return write_pajek(filename, network, labels, node_attributes, edge_attributes)


def read_vertices(lines):
    coordinates = np.zeros((len(lines), 2))
    labels = np.full((len(lines)), None, dtype=object)
    attrs = [{} for _ in lines]
    id_idx = {}
    for i, line in enumerate(lines):
        node_id, *parts = shlex.split(line)
        if not parts:
            label = str(node_id)
        else:
            if len(parts) < 3:
                label = parts[0]
            else:
                try:  # The format specification was never set in stone, it seems
                    label, x, y, *shape = parts
                    x, y = float(x), float(y)
                    coordinates[i, :2] = (x, y)
                    # skip "ellipse" and similar shape descriptors
                    attrs[i] = dict(zip(shape[1::2], shape[2::2]))
                except (TypeError, ValueError):
                    if x is not None:
                        label = line.strip().split(maxsplit=1)[1]
        id_idx[node_id] = i
        labels[i] = label
    if np.sum(np.abs(coordinates)) == 0:
        coordinates = None
    return id_idx, labels, coordinates, attrs


import math

def is_number(s):
    try:
        return math.isfinite(float(s))
    except ValueError:
        return False

def read_edges(id_idx, lines, nvertices):
    def fake_data(x, n):
        values = np.lib.stride_tricks.as_strided(x, (n, ), (0,))
        values.flags.writeable = False
        return values

    values = [np.nan] * len(lines)
    attrs = [{} for _ in lines]
    lines = sorted(
        (id_idx[v1], id_idx[v2], parts)
        for v1, v2, *parts in (shlex.split(line) for line in lines))
    for i, (v1, v2, parts) in enumerate(lines):
        if len(parts) % 2 == 1:
            values[i] = parts[0]
            parts = parts[1:]
        attrs[i] = dict(zip(parts[::2], parts[1::2]))
    v1s, v2s, *_ = zip(*lines)
    try:
        values = np.array(values, dtype=float)
        no_values = np.all(np.isnan(values))
        if no_values:
            values = fake_data(np.array(1.), len(values))
        elif values.size and np.all(values == values[0]):
            values = fake_data(values[0], len(values))
        else:
            values[np.isnan(values)] = 1
        labels = None
    except (TypeError, ValueError):
        labels = values
        values = fake_data(np.array(1.), len(v1s))
    if any(attrs):
        edge_data = table_from_dicts(
            attrs, label_name="edge label", label_values=labels)
    else:
        if labels is not None:
            edge_data = np.array(labels, dtype=object)
        elif not no_values:
            edge_data = values
        else:
            edge_data = None
    return (sp.coo_matrix((values, (np.array(v1s), np.array(v2s))),
                          shape=(nvertices, nvertices)),
            edge_data)


def table_from_dicts(dicts, label_name=None, label_values=None):
    variables = defaultdict(set)
    for dict_ in dicts:
        for key, value in dict_.items():
            variables[key].add(value)
    if label_values is not None:
        label_name = get_unique_names(list(variables), label_name)
        variables[label_name] = set(label_values)
    attrs = []
    attr_indices = {}
    metas = []
    meta_indices = {}
    for key, values in variables.items():
        if all(value == "?" or value == None for value in values):
            # Ignore the variable
            continue
        if all(value == "?" or value == None or is_number(value) for value in values):
            attr_indices[key] = len(attrs)
            attrs.append(ContinuousVariable(key))
        elif is_discrete_values(values):
            attr_indices[key] = len(attrs)
            attrs.append(DiscreteVariable(key, values=natural_sorted(values)))
        else:
            meta_indices[key] = len(metas)
            metas.append(StringVariable(key))
    x = np.full((len(dicts), len(attrs)), np.nan)
    m = np.full((len(dicts), len(metas)), "", dtype=object)
    for i, dict_ in enumerate(dicts):
        for key, value in chain(dict_.items(),
                                ((label_name, label_values[i]), ) if label_values is not None else ()):
            index = attr_indices.get(key)
            if index is not None:
                if value == "?":
                    continue
                elif attrs[index].is_continuous:
                    x[i, attr_indices[key]] = float(value)
                else:
                    x[i, attr_indices[key]] = attrs[index].to_val(value)
            elif key in meta_indices:
                m[i, meta_indices[key]] = value
            # else all values were "?" or None, so we ignored the variable entirely
    domain = Domain(attrs, metas=metas)
    return Table.from_numpy(domain, x, metas=m)


def dict_rows_from_table(data, exclude=None):
    def esc(s):
        if " " in s or '"' in s:
            return '"' + s.replace('"', '\\"') + '"'
        else:
            return s

    all_attrs = [attr
                 for attr in chain(data.domain.attributes, data.domain.metas)
                 if attr.name != exclude]
    return [
        " ".join(
            f"{esc(attr.name)} {esc(attr.str_val(value))}"
            for value, attr in ((data[i, attr], attr) for attr in all_attrs)
            if (value != "" if isinstance(value, str) else not isnan(value))
        )
        for i, inst in enumerate(data)]


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

    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    lines = np.array([line for line in lines if line and line[0] != "%"])
    part_starts = [(line_no, line)
                   for line_no, line in enumerate(lines)
                   if line[0] == "*"]
    network_name = None
    edges = []
    labels = coordinates = attributes = None
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
            id_idx, labels, coordinates, attributes = read_vertices(line_part)
            part_args = part_args.split()
            if len(part_args) > 1:
                in_first_mode = int(part_args[1])
        elif part_type in ("*edges", "*arcs"):
            check_has_vertices()
            edges.append(
                EdgeType[part_type=="*arcs"](
                    *read_edges(id_idx, line_part, len(labels)),
                    name=part_args.strip() or part_type[1:]))
        elif part_type in ("*edgeslist", "*arcslist"):
            check_has_vertices()
            edges.append(
                EdgeType[part_type=="*arcslist"](
                    read_edges_list(id_idx, line_part, len(labels)),
                    name=part_args.strip() or part_type[1:]))
    if np.any(attributes):
        labels = table_from_dicts(attributes, label_name="node label", label_values=labels)
    network = Network(labels, edges, network_name or "", coordinates)
    if in_first_mode is not None:
        network.in_first_mode = in_first_mode
    return network


def write_pajek(path,
                network,
                labels: Optional[list[str]] = None,
                node_attributes: Optional[list[str]] = None,
                edge_attributes: Optional[list[str]] = None):
    if len(network.edges) > 1:
        raise TypeError(
            "This implementation of Pajek format does not support saving "
            "networks with multiple edge types.")
    f = open(path, "wt", encoding="utf-8") if isinstance(path, str) else path
    f.write(f'*Network "{network.name}"\n')
    _write_vertices(f, network, labels, node_attributes)
    if network.edges:
        _write_edges(f, network, edge_attributes)
    if f is not path:
        f.close()


def _write_vertices(f, network,
                    labels: Optional[list[str]] = None,
                    node_attributes: Optional[list[str]] = None):
    if labels is None:
        labels = network.nodes

    f.write(f"*Vertices\t{network.number_of_nodes()}\n")
    if network.coordinates is not None:
        coords = (f"{''.join(f' {c:.4f}' for c in coords)}"
                  for coords in network.coordinates)
    else:
        coords = ()
    if node_attributes is None:
        node_attributes = ()
    for i, label, coordinates, attribute \
            in zip(count(start=1),
                   labels,
                   chain(coords, repeat("0 0" if node_attributes else "")),
                   chain(node_attributes, repeat(()))):
        f.write(f'{i:6} "{label}"\t{coordinates}'
                + (f" ellipse {attribute}" if attribute else "")
                + "\n"
                )


def _write_edges(f, network, edge_attributes: Optional[list[str]] = None):
    edges = network.edges[0]
    f.write("*Arcs\n" if edges.directed else "*Edges\n")
    mat = edges.edges
    if np.all(mat.data == 1) and edge_attributes is None:
        for row, rb, re in zip(count(), mat.indptr, mat.indptr[1:]):
            f.write("".join(f"{row + 1:6} {col + 1:6}\n"
                            for col in mat.indices[rb:re]))
        return
    if edge_attributes is None:
        edge_attributes = [()] * len(mat.data)
    for row, rb, re in zip(count(), mat.indptr, mat.indptr[1:]):
        f.write("".join(f"{row + 1:6} {mat.indices[i] + 1:6} "
                        f"{mat.data[i]:.6f}" +
                        (f" {edge_attributes[i]}"
                         if i < len(edge_attributes) and edge_attributes[i]
                         else "")
                        + "\n"
                        for i in range(rb, re)))


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
