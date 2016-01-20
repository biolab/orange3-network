"""
.. index:: Reading and Writing Networks

.. index::
   single: Network; Reading and Writing Networks

****************************
Reading and Writing Networks
****************************

Use methods in the :obj:`Orange.network.readwrite` module to read and write
networks in Orange. If you must use the original NetworkX read/write methods
(for some reason), do not forget to cast the network object (see
Orange.network.readwrite._wrap method).

"""

import os
import itertools
import tempfile
import gzip

import networkx as nx
import networkx.readwrite.pajek as rwpajek
import networkx.readwrite.gml as rwgml
import networkx.readwrite.gpickle as rwgpickle

import Orange

from .network import Graph, DiGraph, MultiGraph, MultiDiGraph


SUPPORTED_READ_EXTENSIONS = ['.net', '.pajek', '.gml', '.gpickle', '.gz', '.edgelist']
SUPPORTED_WRITE_EXTENSIONS = ['.net', '.pajek', '.gml', '.gpickle', '.edgelist']


def _wrap(g):
    for base, new in [(nx.Graph, Graph),
                      (nx.DiGraph, DiGraph),
                      (nx.MultiGraph, MultiGraph),
                      (nx.MultiDiGraph, MultiDiGraph)]:
        if isinstance(g, base):
            return g if isinstance(g, new) else new(g, name=g.name)
    return g

def _add_doc(myclass, nxclass):
    tmp = nxclass.__doc__.replace('nx.write', 'Orange.network.readwrite.write')
    tmp = tmp.replace('nx.read', 'Orange.network.readwrite.read')
    tmp = tmp.replace('nx', 'Orange.network.nx')
    myclass.__doc__ += tmp

def _is_string_like(obj): # from John Hunter, types-free version
    """Check if obj is string."""
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True

def _get_fh(path, mode='r'):
    """Return a file handle for given path.

    Path can be a string or a file handle.

    Attempt to uncompress/compress files ending in '.gz' and '.bz2'.

    """
    if _is_string_like(path):
        if path.endswith('.gz'):
            import gzip
            fh = gzip.open(path, mode=mode)
        elif path.endswith('.bz2'):
            import bz2
            fh = bz2.BZ2File(path, mode=mode)
        else:
            fh = open(path, mode=mode)
    elif hasattr(path, 'read'):
        fh = path
    else:
        raise ValueError('path must be a string or file handle')
    return fh

def _make_str(t):
    """Return the string representation of t."""
    if _is_string_like(t): return t
    return str(t)

def _check_network_dir(p):
    if type(p) == str:
        if not os.path.isfile(p):
            path = Orange.data.io.find_file(os.path.split(p)[1])

            if os.path.isfile(path):
                return path

            raise OSError('File %s does not exist.' % p)

    return p

def graph_to_table(G):
    """Builds a Data Table from node values."""
    if G.number_of_nodes() > 0:
        features = list(set(itertools.chain.from_iterable(node.keys() for node in G.node.values())))
        data = [[node.get(f).replace('\t', ' ') if isinstance(node.get(f, 1), str) else str(node.get(f, '?'))
                 for f in features]
                for node in G.node.values()]
        fp = tempfile.NamedTemporaryFile('wt', suffix='.tab', delete=False)
        fp.write('\n'.join('\t'.join(line) for line in [features] + data))
        fp.close()
        table = Orange.data.Table(fp.name)
        os.unlink(fp.name)

    return table

def read(path, encoding='UTF-8', auto_table=0):
    """Read graph in any of the supported file formats (.gpickle, .net, .gml).
    The parser is chosen based on the file extension.

    :param path: File or filename to write.
    :type path: string

    Return the network of type :obj:`Orange.network.Graph`,
    :obj:`Orange.network.DiGraph`, :obj:`Orange.network.Graph` or
    :obj:`Orange.network.DiGraph`.

    """
    path = _check_network_dir(path)
    _, ext = os.path.splitext(path.lower())
    if not ext in SUPPORTED_READ_EXTENSIONS:
        raise ValueError('Extension %s is not supported.' % ext)

    if ext in ('.net', '.pajek'):
        return read_pajek(path, encoding, auto_table=auto_table)

    if ext == '.edgelist':
        return read_edgelist(path, auto_table)

    if ext == '.gml':
        return read_gml(path, encoding, auto_table=auto_table)

    if ext == '.gpickle':
        return read_gpickle(path, auto_table=auto_table)

    if ext == '.gz' and path[-6:] == 'txt.gz':
        return read_txtgz(path)

def write(G, path, encoding='UTF-8'):
    """Write graph in any of the supported file formats (.gpickle, .net, .gml).
    The file format is chosen based on the file extension.

    :param G: A Orange graph.
    :type G: Orange.network.Graph

    :param path: File or filename to write.
    :type path: string

    """
    _, ext = os.path.splitext(path.lower())
    if not ext in SUPPORTED_WRITE_EXTENSIONS:
        raise ValueError('Extension %s is not supported. Use %s.' % (ext, ', '.join(supported)))

    if ext in ('.net', '.pajek'):
        write_pajek(G, path, encoding)

    if ext == '.edgelist':
        write_edgelist(G, path)

    if ext == '.gml':
        write_gml(G, path)

    if ext == '.gpickle':
        write_gpickle(G, path)

    if G.items() is not None:
        G.items().save(root + '_items.tab')

    if G.links() is not None:
        G.links().save(root + '_links.tab')


def read_edgelist(path, auto_table):
    G = _wrap(nx.read_edgelist(path))
    if auto_table:
        G.set_items(graph_to_table(G))
    return G


def write_edgelist(G, path):
    nx.write_edgelist(G, path)


def read_gpickle(path, auto_table=False):
    """NetworkX read_gpickle method and wrap graph to Orange network.

    """

    path = _check_network_dir(path)

    G = _wrap(rwgpickle.read_gpickle(path))
    if auto_table:
        G.set_items(graph_to_table(G))
    return G

#~ _add_doc(read_gpickle, rwgpickle.read_gpickle)

def write_gpickle(G, path):
    """NetworkX write_gpickle method.

    """

    rwgpickle.write_gpickle(G, path)

_add_doc(write_gpickle, rwgpickle.write_gpickle)

def read_pajek(path, encoding='UTF-8', project=False, auto_table=False):
    """Reimplemented method for reading Pajek files; written in
    C++ for maximum performance.

    :param path: File or file name to write.
    :type path: string

    :param encoding: Encoding of input text file, default 'UTF-8'.
    :type encoding: string

    :param project: Determines whether the input file is a Pajek project file,
        possibly containing multiple networks and other data. If :obj:`True`,
        a list of networks is returned instead of just a network. Default is
        :obj:`False`.
    :type project: boolean.

    Return the network (or a list of networks if project=:obj:`True`) of type
    :obj:`Orange.network.Graph` or :obj:`Orange.network.DiGraph`.


    Examples

    >>> G=Orange.network.nx.path_graph(4)
    >>> Orange.network.readwrite.write_pajek(G, "test.net")
    >>> G=Orange.network.readwrite.read_pajek("test.net")

    To create a Graph instead of a MultiGraph use

    >>> G1=Orange.network.Graph(G)

    References

    See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm
    for format information.

    """

    path = _check_network_dir(path)
    G = _wrap(rwpajek.read_pajek(path))

    # Additionally read values into Table; needed to get G nodes properly sorted
    # (Consult OWNxFile.readDataFile(), orangeom.GraphLayout.readPajek(), and the Pajek format spec)
    import shlex, numpy as np
    rows, metas, remapping = [], [], {}
    with open(path) as f:
        for line in f:
            if line.lower().startswith('*vertices'):
                nvertices = int(line.split()[1])
                break
        # Read vertices lines
        for line in f:
            parts = shlex.split(line)[:4]
            if len(parts) == 1:
                i = label = parts[0]
            elif len(parts) == 2:
                i, label = parts
                metas.append((label,))
            elif len(parts) == 4:
                i, label, x, y = parts
                # The format specification was never set in stone, it seems
                try:
                    x, y = float(x), float(y)
                except ValueError:
                    metas.append((label, x, y))
                else:
                    rows.append((x, y))
                    metas.append((label,))
            i = int(i) - 1  # -1 because pajek is 1-indexed
            remapping[label] = i
            nvertices -= 1
            if not nvertices: break
    from Orange.data import Domain, Table, ContinuousVariable, StringVariable
    # Construct x-y-label table (added in OWNxFile.readDataFile())
    table = None
    vars = [ContinuousVariable('x'), ContinuousVariable('y')] if rows else []
    meta_vars = [StringVariable('label ' + str(i)) for i in range(len(metas[0]) if metas else 0)]
    if rows or metas:
        domain = Domain(vars, metas=meta_vars)
        table = Table.from_numpy(domain,
                                 np.array(rows, dtype=float).reshape(len(metas),
                                                                     len(rows[0]) if rows else 0),
                                 metas=np.array(metas, dtype=str))
    if table is not None and auto_table:
        G.set_items(table)
    # Relabel nodes to integers, sorted by appearance
    for node in G.node:
        G.node[node]['label'] = node
    nx.relabel_nodes(G, remapping, copy=False)
    assert not table or len(table) == G.number_of_nodes(), 'There was a bug in NetworkX. Please update to git if need be'
    return G

def write_pajek(G, path, encoding='UTF-8'):
    """A copy & paste of NetworkX's function with some bugs fixed (call the new
    generate_pajek).

    """

    fh=_get_fh(path, 'wb')
    for line in generate_pajek(G):
        line+='\n'
        fh.write(line.encode(encoding))

_add_doc(write_pajek, rwpajek.write_pajek)

def parse_pajek(lines):
    """Parse string in Pajek file format. See read_pajek for usage examples.

    :param lines: a string of network data in Pajek file format.
    :type lines: string

    """

    return read_pajek(lines)


def generate_pajek(G):
    """A copy & paste of NetworkX's function with some bugs fixed (generate
    one line per object: vertex, edge, arc. Do not add one per entry in data
    dictionary).

    Generate lines in Pajek graph format.

    :param G: A Orange graph.
    :type G: Orange.network.Graph

    References

    See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm
    for format information.

    """

    if G.name=='':
        name='NetworkX'
    else:
        name=G.name
    yield '*network %s'%name

    # write nodes with attributes
    yield '*vertices %s'%(G.order())
    nodes = G.nodes()
    # make dictionary mapping nodes to integers
    nodenumber=dict(zip(nodes,range(1,len(nodes)+1)))
    for n in nodes:
        na=G.node.get(n,{})
        x=na.get('x',0.0)
        y=na.get('y',0.0)
        id=int(na.get('id',nodenumber[n]))
        nodenumber[n]=id
        shape=na.get('shape','ellipse')
        s = ' '.join(map(_make_str,(id,n,x,y,shape)))
        for k,v in na.items():
            if k != 'x' and k != 'y':
                s += ' %s %s'%(k,v)
        yield s

    # write edges with attributes
    if G.is_directed():
        yield '*arcs'
    else:
        yield '*edges'
    for u,v,edgedata in G.edges(data=True):
        d=edgedata.copy()
        value=d.pop('weight',1.0) # use 1 as default edge value
        s = ' '.join(map(_make_str,(nodenumber[u],nodenumber[v],value)))
        for k,v in d.items():
            if not _is_string_like(v):
                v = repr(v)
            # add quotes to any values with a blank space
            if " " in v:
                v="\"%s\"" % v.replace('"', r'\"')
            s += ' %s %s'%(k,v)
        yield s


#_add_doc(generate_pajek, rwpajek.generate_pajek)

def read_gml(path, encoding='latin-1', relabel=False, auto_table=False):
    """NetworkX read_gml method and wrap graph to Orange network.

    """

    path = _check_network_dir(path)

    G = _wrap(rwgml.read_gml(path, encoding, relabel))
    if auto_table:
        G.set_items(graph_to_table(G))
    return G

_add_doc(read_gml, rwgml.read_gml)

def write_gml(G, path):
    """NetworkX write_gml method.

    """

    rwgml.write_gml(G, path)

def read_txtgz(path):
    f = gzip.open(path, 'rb')
    content = f.read()
    f.close()

    content = content.splitlines()
    comments = (line for line in content if line.strip().startswith(b'#'))
    content = (line for line in content if not line.strip().startswith(b'#'))

    if b"directed graph" in b''.join(comments).lower():
        G = DiGraph()
    else:
        G = Graph()

    G.add_edges_from([int(node) for node in coors.strip().split(b'\t')]
                      for coors in content
                      if len(coors.strip().split(b'\t')) == 2)

    return G

_add_doc(write_gml, rwgml.write_gml)
