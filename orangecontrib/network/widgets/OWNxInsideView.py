
import Orange
from Orange.widgets import gui, widget, settings
import orangecontrib.network as network


class NxInsideView(network.NxView):
    """Network Inside View

    """

    def __init__(self, nhops):
        network.NxView.__init__(self)

        self._nhops = nhops
        self._center_node = None

    def init_network(self, graph):
        self._network = graph

        if graph is None:
            return None

        selected = self._nx_explorer.networkCanvas.selected_nodes()
        if selected is None or len(selected) <= 0:
            self._center_node = next(graph.nodes_iter())
        else:
            self._center_node = selected[0]

        nodes = self._get_neighbors()
        return network.nx.Graph.subgraph(self._network, nodes)

    def update_network(self):
        nodes = self._get_neighbors()
        subnet = network.nx.Graph.subgraph(self._network, nodes)

        if self._nx_explorer is not None:
            self._nx_explorer.change_graph(subnet)

    def set_nhops(self, nhops):
        self._nhops = nhops

    def node_selection_changed(self):
        selection = self._nx_explorer.networkCanvas.selected_nodes()
        if len(selection) == 1:
            self._center_node = selection[0]
            self.update_network()

    def _get_neighbors(self):
        nodes = set([self._center_node])
        for n in range(self._nhops):
            neighbors = set()
            for node in nodes:
                neighbors.update(self._network.neighbors(node))
            nodes.update(neighbors)
        return nodes

class OWNxInsideView(widget.OWWidget):
    name = "Network Inside View"
    description = "Orange widget for community detection in networks"
    icon = "icons/NetworkInsideView.svg"
    priority = 6460

    resizing_enabled = False

    outputs = [("Nx View", network.NxView)]

    _nhops = settings.Setting(2)

    def __init__(self):
        super().__init__()

        self._nhops = 2

        ib = gui.widgetBox(self.controlArea, "Preferences", orientation="vertical")
        gui.spin(ib, self, "_nhops", 1, 6, 1, label="Number of hops: ", callback=self.update_view)

        self.inside_view = NxInsideView(self._nhops)
        self.send("Nx View", self.inside_view)

        self.warning('This widget, at best, does nothing at the moment. Check back later.')

    def update_view(self):
        self.inside_view.set_nhops(self._nhops)

        self.inside_view.update_network()
