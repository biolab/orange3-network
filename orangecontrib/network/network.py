"""
.. index:: Network

*********
BaseGraph
*********

BaseGraph provides methods to work with additional data--describing nodes and
edges:

* items (:obj:`Orange.data.Table`) - information on nodes. Each row in the table corresponds to a node with ID matching row index.
* links (:obj:`Orange.data.Table`) - information on edges. Each row in the table corresponds to an edge. Two columns titled "u" and "v" should be specified in the table which contain indices of nodes on the given edge.

The BaseGraph class contains also other methods that are common to the four graph types.

.. autoclass:: Orange.network.BaseGraph
   :members:

***********
Graph Types
***********

The reference in this section is complemented with the original NetworkX
library reference. For a complete documentation please refer to the
`NetworkX docs <http://networkx.lanl.gov/reference/>`_. All methods from the
NetworkX package can be used for graph analysis and manipulation. For reading
and writing graphs refer to the Orange.network.readwrite docs.

Graph
=====

.. autoclass:: Orange.network.Graph
   :members:

DiGraph
=======

.. autoclass:: Orange.network.DiGraph
   :members:

MultiGraph
==========

.. autoclass:: Orange.network.MultiGraph
   :members:

MultiDiGraph
============

.. autoclass:: Orange.network.MultiDiGraph
   :members:

"""

import copy
import math
from itertools import chain

import numpy as np
import networkx as nx

import Orange

class MdsTypeClass():
    def __init__(self):
        self.componentMDS = 0
        self.exactSimulation = 1
        self.MDS = 2

MdsType = MdsTypeClass()

def _get_doc(doc):
    return doc.replace('nx.', 'Orange.network.') if doc else ''

class BaseGraph():
    """A collection of methods inherited by all graph types (:obj:`Graph`,
    :obj:`DiGraph`, :obj:`MultiGraph` and :obj:`MultiDiGraph`).

    """

    def __init__(self):
        self._items = None
        self._links = None

    def items(self):
        """Return the :obj:`Orange.data.Table` items with data about network
        nodes.

        """

        if self._items is not None and \
                        len(self._items) != self.number_of_nodes():
            print("Warning: items length does not match the number of nodes.")

        return self._items

    def set_items(self, items=None):
        """Set the :obj:`Orange.data.Table` items to the given data. Notice
        that the number of instances must match the number of nodes.

        """

        if items is not None:
            if not isinstance(items, Orange.data.Table):
                raise TypeError('items must be of type \'Orange.data.Table\'')
            if len(items) != self.number_of_nodes():
                print("Warning: items length must match the number of nodes.")

        self._items = items

    def links(self):
        """Return the :obj:`Orange.data.Table` links with data about network
        edges.

        """

        if self._links is not None \
                    and len(self._links) != self.number_of_edges():
            print("Warning: links length does not match the number of edges.")

        return self._links

    def set_links(self, links=None):
        """Set the :obj:`Orange.data.Table` links to the given data. Notice
        that the number of instances must match the number of edges.

        """

        if links is not None:
            if not isinstance(links, Orange.data.Table):
                raise TypeError('links must be of type \'Orange.data.Table\'')
            if len(links) != self.number_of_edges():
                print("Warning: links length must match the number of edges.")

        self._links = links

    def to_orange_network(self):
        """Convert the current network to >>Orange<< NetworkX standard. To use
        :obj:`Orange.network` in Orange widgets, set node IDs to be range
        [0, no_of_nodes - 1].

        """

        G = self.__class__()
        node_list = sorted(self.nodes())
        node_to_index = dict(zip(node_list, range(self.number_of_nodes())))
        index_to_node = dict(zip(range(self.number_of_nodes()), node_list))

        G.add_nodes_from(zip(range(self.number_of_nodes()), [copy.deepcopy(self.node[nid]) for nid in node_list]))
        G.add_edges_from(((node_to_index[u], node_to_index[v], copy.deepcopy(self.edge[u][v])) for u, v in self.edges()))

        for id in G.node.keys():
            G.node[id]['old_id'] = index_to_node[id]

        if self.items():
            G.set_items(self.items())

        if self.links():
            G.set_links(self.links())

        return G

    ### TODO: OVERRIDE METHODS THAT CHANGE GRAPH STRUCTURE, add warning prints

    def items_vars(self):
        """Return a list of features in the :obj:`Orange.data.Table` items."""
        if not self._items: return []
        return list(chain(self._items.domain.variables,
                          self._items.domain.metas))

    def links_vars(self):
        """Return a list of features in the :obj:`Orange.data.Table` links."""
        if not self._links: return []
        return [i for i in chain(self._items.domain.variables,
                                 self._items.domain.metas)
                if i.name not in ('u', 'v')]

    def subgraph(self, nbunch):
        G = super().subgraph(nbunch)
        G = G.to_orange_network()
        if self.items():
            items = self.items()[sorted(G.nodes()), :]
            G.set_items(items)
        return G

class Graph(BaseGraph, nx.Graph):
    """Bases: `NetworkX.Graph <http://networkx.lanl.gov/reference/classes.graph.html>`_,
    :obj:`Orange.network.BaseGraph`

    """

    def __init__(self, data=None, name='', **attr):
        nx.Graph.__init__(self, data, name=name, **attr)
        BaseGraph.__init__(self)
        # TODO: _links

    __doc__ += _get_doc(nx.Graph.__doc__)
    __init__.__doc__ = _get_doc(nx.Graph.__init__.__doc__)

class DiGraph(BaseGraph, nx.DiGraph):
    """Bases: `NetworkX.DiGraph <http://networkx.lanl.gov/reference/classes.digraph.html>`_,
    :obj:`Orange.network.BaseGraph`

    """


    def __init__(self, data=None, name='', **attr):
        nx.DiGraph.__init__(self, data, name=name, **attr)
        BaseGraph.__init__(self)

    __doc__ += _get_doc(nx.DiGraph.__doc__)
    __init__.__doc__ = _get_doc(nx.DiGraph.__init__.__doc__)

class MultiGraph(BaseGraph, nx.MultiGraph):
    """Bases: `NetworkX.MultiGraph <http://networkx.lanl.gov/reference/classes.multigraph.html>`_,
    :obj:`Orange.network.BaseGraph`

    """


    def __init__(self, data=None, name='', **attr):
        nx.MultiGraph.__init__(self, data, name=name, **attr)
        BaseGraph.__init__(self)

    __doc__ += _get_doc(nx.MultiGraph.__doc__)
    __init__.__doc__ = _get_doc(nx.MultiGraph.__init__.__doc__)

class MultiDiGraph(BaseGraph, nx.MultiDiGraph):
    """Bases: `NetworkX.MultiDiGraph <http://networkx.lanl.gov/reference/classes.multidigraph.html>`_,
    :obj:`Orange.network.BaseGraph`

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
