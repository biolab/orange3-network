============
Network File
============

.. figure:: icons/network-file.png

Constructs network from atrribute-valued data.

Signals
-------

**Inputs**:

-  (None)

**Outputs**:

-  **Network**

   A built network of nodes and edges.

-  **Items**

   A table of network data.

Description
-----------

**Network File** reads the input data (data on nodes and edges) and constructs a :ref:`Network` instance from it. Both the constructed network and the network data are sent to the output channel.

History of the most recently opened files is maintained in the widget.
The widget also includes a directory with sample network data that come with Orange3-Networks.

The widget reads `NetworkX <https://networkx.github.io/>`_ data (.gpickle and .edgelist), `GML <https://en.wikipedia.org/wiki/Graph_Modelling_Language>`_ (.gml) and `Pajek <http://mrvar.fdv.uni-lj.si/pajek/>`_ (.net, .pajek).  

.. figure:: images/network-stamped.png

1. Graph File: builds a network graph from the data. If '*Build graph data table automatically*' is checked, graph will be constructed automatically.

2. Vertices Data File: information on vertices (labels, features). Select *(none)* to skip loading vertices file.

3. Information on the network (type of graph, number of nodes, number of edges, features).

Example
-------

A simple example to load the data is to click a dropdown menu below *Graph File* and select *Browse documentation data sets*. This will take you to the folder with pre-lodaded data. Select *lastfm.net*. **Network File** will load the data and find the corresponding vertices data file automatically. You can see additional information on the data under *Info*.

.. figure:: images/network-example.png

To view the constructed network, use :doc:`Network Explorer <networkexplorer>`.

To load your own data, use the *folder* icon next to the drop down menu. Select the file you want to load from the computer.
