"""
.. index:: Network

*********
BaseGraph
*********

BaseGraph provides methods to work with additional data--describing nodes and
edges:

* items (:obj:`Table`) - information on nodes. Each row in the table corresponds to a node with ID matching row index.
* links (:obj:`Table`) - information on edges. Each row in the table corresponds to an edge. Two columns titled "u" and "v" should be specified in the table which contain indices of nodes on the given edge.

The BaseGraph class contains also other methods that are common to the four graph types.

.. autoclass:: orangecontrib.network.BaseGraph
   :members:

***********
Graph Types
***********

The reference in this section is complemented with the original NetworkX
library reference. For a complete documentation please refer to the
`NetworkX docs <http://networkx.lanl.gov/reference/>`_. All methods from the
NetworkX package can be used for graph analysis and manipulation. For reading
and writing graphs refer to the orangecontrib.network.readwrite docs.

Graph
=====

.. autoclass:: orangecontrib.network.Graph
   :members:

DiGraph
=======

.. autoclass:: orangecontrib.network.DiGraph
   :members:

MultiGraph
==========

.. autoclass:: orangecontrib.network.MultiGraph
   :members:

MultiDiGraph
============

.. autoclass:: orangecontrib.network.MultiDiGraph
   :members:

"""

import copy
from itertools import chain

import networkx as nx

from Orange.data import Table


class MdsTypeClass():
    def __init__(self):
        self.componentMDS = 0
        self.exactSimulation = 1
        self.MDS = 2


MdsType = MdsTypeClass()


def _get_doc(doc):
    return doc.replace('nx.', 'orangecontrib.network.') if doc else ''


class BaseGraph():
    """A collection of methods inherited by all graph types (:obj:`Graph`,
    :obj:`DiGraph`, :obj:`MultiGraph` and :obj:`MultiDiGraph`).
    """
    def __init__(self):
        self._items = None
        self._links = None

    def items(self):
        """Return the :obj:`Table` items with data about network nodes.
        """
        if self._items is not None and \
                        len(self._items) != self.number_of_nodes():
            print("Warning: items length does not match the number of nodes.")

        return self._items

    def set_items(self, items=None):
        """Set the :obj:`Table` items to the given data. Notice
        that the number of instances must match the number of nodes.
        """
        if items is not None:
            if not isinstance(items, Table):
                raise TypeError('items must be of type \'Table\'')
            if len(items) != self.number_of_nodes():
                print("Warning: items length must match the number of nodes.")

        self._items = items

    def links(self):
        """Return the :obj:`Table` links with data about network edges.
        """
        if self._links is not None \
                    and len(self._links) != self.number_of_edges():
            print("Warning: links length does not match the number of edges.")

        return self._links

    def set_links(self, links=None):
        """Set the :obj:`Table` links to the given data. Notice
        that the number of instances must match the number of edges.
        """
        if links is not None:
            if not isinstance(links, Table):
                raise TypeError('links must be of type \'Table\'')
            if len(links) != self.number_of_edges():
                print("Warning: links length must match the number of edges.")

        self._links = links

    def to_orange_network(self):
        """Convert the current network to >>Orange<< NetworkX standard. To use
        :obj:`orangecontrib.network` in Orange widgets, set node IDs to be range
        [0, no_of_nodes - 1].
        """
        G = self.__class__()
        node_list = sorted(self.nodes())
        node_to_index = dict(zip(node_list, range(self.number_of_nodes())))
        index_to_node = dict(zip(range(self.number_of_nodes()), node_list))

        G.add_nodes_from(zip(range(self.number_of_nodes()), [copy.deepcopy(self.node[nid]) for nid in node_list]))
        G.add_edges_from(((node_to_index[u], node_to_index[v], copy.deepcopy(self.adj[u][v])) for u, v in self.edges()))

        for id, data in G.nodes(data=True):
            data['old_id'] = index_to_node[id]

        if self.items():
            G.set_items(self.items())

        if self.links():
            G.set_links(self.links())

        return G

    ### TODO: OVERRIDE METHODS THAT CHANGE GRAPH STRUCTURE, add warning prints

    def items_vars(self):
        """Return a list of features in the :obj:`Table` items."""
        if not self._items: return []
        return list(chain(self._items.domain.variables,
                          self._items.domain.metas))

    def links_vars(self):
        """Return a list of features in the :obj:`Table` links."""
        if not self._links: return []
        return [i for i in chain(self._items.domain.variables,
                                 self._items.domain.metas)
                if i.name not in ('u', 'v')]

    def subgraph(self, nbunch):
        G = self.copy()
        G.remove_nodes_from(list(set(G) - set(nbunch)))
        G = G.to_orange_network()
        if self.items():
            items = self.items()[sorted(G.nodes()), :]
            G.set_items(items)
        return G

    def copy(self, *args, **kwargs):
        obj = super().copy(*args, **kwargs)
        obj._items = copy.deepcopy(self._items)
        obj._links = copy.deepcopy(self._links)
        return obj

    @classmethod
    def fresh_copy(cls):
        return cls()


class Graph(BaseGraph, nx.Graph):
    """Bases: `NetworkX.Graph <http://networkx.lanl.gov/reference/classes.graph.html>`_,
    :obj:`orangecontrib.network.BaseGraph`
    """
    def __init__(self, data=None, name='', **attr):
        nx.Graph.__init__(self, data, name=name, **attr)
        BaseGraph.__init__(self)
        # TODO: _links

    __doc__ += _get_doc(nx.Graph.__doc__)
    __init__.__doc__ = _get_doc(nx.Graph.__init__.__doc__)


class DiGraph(BaseGraph, nx.DiGraph):
    """Bases: `NetworkX.DiGraph <http://networkx.lanl.gov/reference/classes.digraph.html>`_,
    :obj:`orangecontrib.network.BaseGraph`
    """
    def __init__(self, data=None, name='', **attr):
        nx.DiGraph.__init__(self, data, name=name, **attr)
        BaseGraph.__init__(self)

    __doc__ += _get_doc(nx.DiGraph.__doc__)
    __init__.__doc__ = _get_doc(nx.DiGraph.__init__.__doc__)


class MultiGraph(BaseGraph, nx.MultiGraph):
    """Bases: `NetworkX.MultiGraph <http://networkx.lanl.gov/reference/classes.multigraph.html>`_,
    :obj:`orangecontrib.network.BaseGraph`
    """


    def __init__(self, data=None, name='', **attr):
        nx.MultiGraph.__init__(self, data, name=name, **attr)
        BaseGraph.__init__(self)

    __doc__ += _get_doc(nx.MultiGraph.__doc__)
    __init__.__doc__ = _get_doc(nx.MultiGraph.__init__.__doc__)


class MultiDiGraph(BaseGraph, nx.MultiDiGraph):
    """Bases: `NetworkX.MultiDiGraph <http://networkx.lanl.gov/reference/classes.multidigraph.html>`_,
    :obj:`orangecontrib.network.BaseGraph`
    """
    def __init__(self, data=None, name='', **attr):
        nx.MultiDiGraph.__init__(self, data, name=name, **attr)
        BaseGraph.__init__(self)

    __doc__ += _get_doc(nx.MultiDiGraph.__doc__)
    __init__.__doc__ = _get_doc(nx.MultiDiGraph.__init__.__doc__)


class NxView(object):
    """Network View"""
    def __init__(self, **attr):
        self._network = None
        self._nx_explorer = None

    def set_nx_explorer(self, _nx_explorer):
        self._nx_explorer = _nx_explorer

    def init_network(self, graph):
        return graph

    def node_selection_changed(self):
        pass

    def update_network(self):
        if self._nx_explorer is not None and self._network is not None:
            subnet = self._network
            self._nx_explorer.change_graph(subnet)
