"""
*************************
Network Classes in Orange
*************************

Orange network classes provide methods for graph manipulation,
network analysis, and layout optimization. They are derived from
`NetworkX basic graph types <http://networkx.lanl.gov/reference/classes.html>`_
and :obj:`Orange.network.BaseGraph`.

We support four graph types: :obj:`Orange.network.Graph`,
:obj:`Orange.network.DiGraph`, :obj:`Orange.network.MultiGraph`, and
:obj:`Orange.network.MultiDiGraph`. Choose the graph type that matches the
structure of the graph you wish to represent.

Examples
========

Reading and Writing Networks
----------------------------

Network class reads and writes Pajek (.net), GML, and gpickle file formats.

:download:`network-read-nx.py <code/network-read-nx.py>`:

.. literalinclude:: code/network-read.py
    :lines: 5-6

Visualize Networks in Net Explorer Widget
-----------------------------------------

To display a network in Net Explorer widget write:

part of :download:`network-widget.py <code/network-widget.py>`

.. literalinclude:: code/network-widget.py
    :lines: 10-16

.. image:: files/network-explorer.png
    :width: 100%

"""

from pkg_resources import resource_filename
def networks():
    yield ('', resource_filename(__name__, 'networks'))

import networkx as nx

from .network import *

from . import community
from . import snap
