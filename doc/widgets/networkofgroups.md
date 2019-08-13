Network Generator
=================

.. figure:: icons/network-generator.png

Signals
-------

**Inputs**:

-  (None)

**Outputs**:

-  **Generated Network**

   An instance of Network Graph.

Description
-----------

**Network Generator** constructs exemplary networks. It is mostly intended for teaching/learning about networks.

.. figure:: images/network-generator.png

1. Generate graph:
   - `Balanced tree <https://networkx.github.io/documentation/development/reference/generated/networkx.generators.classic.balanced_tree.html#networkx.generators.classic.balanced_tree>`_
   - `Barbell <https://en.wikipedia.org/wiki/Barbell_graph>`_
   - `Circular ladder <http://mathworld.wolfram.com/CircularLadderGraph.html>`_
   - `Complete <https://en.wikipedia.org/wiki/Complete_graph>`_
   - `Complete bipartite <https://en.wikipedia.org/wiki/Bipartite_graph>`_
   - `Cycle <https://en.wikipedia.org/wiki/Cycle_(graph_theory)>`_
   - `Grid <http://mathworld.wolfram.com/GridGraph.html>`_
   - `Hypercube <https://en.wikipedia.org/wiki/Hypercube_graph>`_
   - `Ladder <https://en.wikipedia.org/wiki/Ladder_graph>`_
   - `Lobster <http://mathworld.wolfram.com/LobsterGraph.html>`_
   - `Lollipop <https://en.wikipedia.org/wiki/Lollipop_graph>`_
   - `Path <https://en.wikipedia.org/wiki/Path_(graph_theory)>`_
   - `Regular <https://en.wikipedia.org/wiki/Regular_graph>`_
   - `Scale-free <https://en.wikipedia.org/wiki/Scale-free_network>`_
   - `Shell <https://networkx.github.io/documentation/development/reference/generated/networkx.generators.random_graphs.random_shell_graph.html#networkx.generators.random_graphs.random_shell_graph>`_
   - `Star <https://en.wikipedia.org/wiki/Star_(graph_theory)>`_
   - `Waxman <https://networkx.github.io/documentation/development/reference/generated/networkx.generators.geometric.waxman_graph.html#networkx.generators.geometric.waxman_graph>`_
   - `Wheel <https://en.wikipedia.org/wiki/Wheel_graph>`_
2. Approx. number of nodes: nodes that should roughly be in the network (some networks cannot exactly satisfy this condition, hence an approximation).
3. If *Auto-generate* is on, the widget will automatically send the constructed graph to the output. Alternatively, press *Generate graph*.

Example
-------

**Network Generator** is a nice tool to explore typical graph structures.

.. figure:: images/network-generator-example.png

Here, we generated a *Scale-free* graph with approximately 50 vertices and sent it to :doc:`Network Analysis <networkanalysis>`. We computed the clustering coefficient and sent the data to :doc:`Network Explorer <networkexplorer>`. Finally, we observed the generated graph in the visualization and set the size of the vertices to *Clustering coefficient*. This is a nice tool to observe and explain the properties of networks.
