================
Network Explorer
================

.. figure:: icons/network-explorer.png

Visualizes the network and allows for an exploratory analysis.

Signals
-------

**Inputs**:

-  **Network**

   A built network of nodes and edges.

-  **Node Subset**

   A subset of vertices.

-  **Node Data**

   Data from a subset of network vertices.

-  **Node Distances**

   Distances from nodes.

**Outputs**:

-  **Selected sub-network**

   Selected network subset.

-  **Distance matrix**

   Distance measures in a distance matrix.

-  **Selected items**

   Data from selected network vertices.

-  **Highlighted items**

   Data from highlighted network vertices.

-  **Remaining items**

   Data from remaining network vertices.

Description
-----------

**Network Explorer** is the main widget for visualizing network graphs. It takes a network on the input and optimizes projection based on a Fruchterman-Reingold layout optimization. Note, however, that nodes can be moved around at will.

The widget can be connected with other widget for further analysis or in-depth exploration of network subsets.

.. figure:: images/networkexplorer-stamped.png

Display
~~~~~~~

1. Info:
   Information on the network on the input. Reported are the number of nodes and edges.

2. Nodes:
   - *Re-layout* re-optimizes network graph with a Fructherman-Reingold optimization.
   - *Color*: color nodes by attributes or class.
   - *Size*: set the size of nodes by attributes or class.
   - *Min* and *Max* denote the smallest and largest node size. Having these two values further apart will result in having a more pronounced difference between nodes.
   - *Invert* will reverse the size of nodes, making the smallest nodes the largest and vice versa.

3. Node labels | tooltips:
   In the left pane, set the labels you want displayed next to nodes.
   In the right pane, select the attributes displayed in tooltips (when hovering over a node).

4. Edges:
   If *Relative edge width* is ticked, edge width will be set to the corresponding weight (the higher the weight, the bolder the edge).
   If *Show edge weights* is ticked, corresponding weights will be displayed on each edge of the network.

Marking
~~~~~~~

1. Info:
   Information on the number of nodes in the network, number of selected and number of highlighted nodes.

2. Highlight nodes:
   - None: nodes aren't highlighted.
   - *... whose attributes contain*: highlight nodes that satisfy a condition (e.g. '2004' will highlight all nodes that have '2004' as value).
   - *... neighbours of selected, N hopes away*: nodes that are N hops away from selected nodes. N is set in the *Hops* spinbox.
   - *... with at least N connections*: highlight nodes with more than a specified number of connections (e.g. all nodes that have 3 or more neighbours).
   - *... with at most N connections*: highlight nodes with equal or less than the specified number of connections (e.g. all nodes that have 3 neighbours or less). N is set in the *Connections* spinbox.
   - *... with more connections than any neighbor*: highlight nodes that are well-connected (find hubs).
   - *... with more connections than average neighbor*: highlight relatively well-connected nodes (nodes that are better connected than most).
   - *... with most connections*: find N nodes that are best connected. N is set in the *Number of nodes* spinbox (defines how many nodes should be highlighted).
   - *... given in the ItemSubset input signal*: if a data subset is provided on the input, the user can select to highlight nodes matching the provided subset by a selected attribute.

3. If *Output Changes Automatically* is ticked, all the data is automatically sent to the output. Alternatively, press *Output changes*. 
Examples
--------

The first example will show a few custom visualizations of the *leu_by_genesets.net* data.

First, we see that our network layout is already optimized. Well-connected nodes lie close to one another, making it easy to see cliques and clusters.

.. figure:: images/networkexplorer-example1.png

By hovering over a node, a tooltip with all the selected data will be displayed.

.. figure:: images/networkexplorer-example2.png

We also set *label 0* as our node label, to see which genes appear where in the table. Labelling can get quite messy, thus selecting node lables is best only when analysing small networks or network subsets.

Now we set also *t-test* to color by and *no. of genesets* for node size. This will show us which genes have statistically significant expression level and which belong to bigger genesets.

.. figure:: images/networkexplorer-example3.png

Now, let's mark the best connected nodes. Go to *Marking* tab and select *Highlight nodes... with at least N connections*. Since our network is small, we've set *Connections* to be 2. This means highlighted nodes will have 2 or more connections each.

Now let's press *Enter*. This is how we select highlighted nodes. Highlighted nodes are marked with orange, while selected become red.

You can view the selection in another **Network Explorer** or observe the properties of this network in a **Data Table**.

.. figure:: images/networkexplorer-example4.png

When connecting to a **Data Table**, a menu *Edit links* will appear. Here you can set, what you want on the output. Selected nodes will be on the output by default, but you can change it to highlighted nodes.
