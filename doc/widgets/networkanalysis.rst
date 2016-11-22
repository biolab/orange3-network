================
Network Analysis
================

.. figure:: icons/network-analysis.png

Statistical analysis of network data.

Signals
-------

**Inputs**:

-  **Network**

   A built network of nodes and edges.

-  **Items**

   Data from network vertices.

**Outputs**:

-  **Network**

   A built network of nodes and edges.

-  **Items**

   Data from network vertices.

Description
-----------

**Network Analysis** widget adds additional information on network graph to the output. It computes a variety of statistical scores at node and graph-level and appends them as features.

.. figure:: images/networkanalysis-stamped.png

1. *Graph-level indices*:
   - Number of nodes: number of vertices in the graph
   - Number of edge: number of connections in the graph
   - Average `degree <https://en.wikipedia.org/wiki/Degree_(graph_theory)>`_: average number of connections per node
   - `Diameter <https://networkx.github.io/documentation/networkx-1.9.1/reference/generated/networkx.algorithms.distance_measures.diameter.html?highlight=diameter#diameter>`_: maximum eccentricity.
   - `Radius <https://networkx.github.io/documentation/networkx-1.9.1/reference/generated/networkx.algorithms.distance_measures.radius.html?highlight=radius#radius>`_: minimum eccentricity.
   - `Average shortest path length <https://networkx.github.io/documentation/networkx-1.9.1/reference/generated/networkx.algorithms.shortest_paths.generic.average_shortest_path_length.html?highlight=average%20shortest%20path%20length#average-shortest-path-length>`_: average shortest path between node pairs.
   - `Density <https://networkx.github.io/documentation/networkx-1.9.1/reference/generated/networkx.classes.function.density.html?highlight=density#density>`_: ratio between the number of edges and number of nodes
   - `Degree assortativity coefficient <https://networkx.github.io/documentation/networkx-1.9.1/reference/generated/networkx.algorithms.assortativity.degree_assortativity_coefficient.html#degree-assortativity-coefficient>`_: similarity of connections in the graph with respect to the node degree.
   - `Degree pearson correlation coefficient <https://networkx.github.io/documentation/networkx-1.9.1/reference/generated/networkx.algorithms.assortativity.degree_pearson_correlation_coefficient.html?highlight=pearson%20correlation#degree-pearson-correlation-coefficient>`_: degree assortativity with pearson r function.
   - Estrada index: 
   - `Graph clique number <https://en.wikipedia.org/wiki/Clique_(graph_theory)>`_: size of the largest clique.
   - Graph number of cliques: number of complete subgraphs in the network.
   - `Graph transitivity <http://www.sci.unich.it/~francesc/teaching/network/transitivity.html>`_: ratio of triangles (3 perfectly connected nodes) to triades (any 3 nodes)
   - Average clustering coefficient:
   - Number of connected components:
   - Number of strongly connected components:
   - Number of weakly connected components:
   - Number of attracting components:

2. *Node-level indices*:
   - 