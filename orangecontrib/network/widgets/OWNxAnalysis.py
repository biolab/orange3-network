import inspect
from collections import namedtuple

import numpy as np
from scipy.sparse import csgraph

from AnyQt.QtCore import QThread, Qt
from AnyQt.QtWidgets import QWidget, QGridLayout

from Orange.data import ContinuousVariable, Table, Domain
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, Msg
from orangecontrib.network.network import Network

NODELEVEL, GRAPHLEVEL, INTERNAL = range(3)
UNDIRECTED, DIRECTED, GENERAL = range(3)

ERRORED = object()
TERMINATED = object()


def shortest_paths_nan_diag(network):
    paths = csgraph.floyd_warshall(
        network.edges[0].edges, network.edges[0].directed).astype(float)
    diag = np.lib.stride_tricks.as_strided(
        paths, (len(paths), ), ((len(paths) + 1) * paths.dtype.itemsize,))
    diag[:] = np.nan
    return paths


def density(network):
    n = network.number_of_nodes()
    if n < 2:
        return 0.0

    num = network.number_of_edges()
    if not network.edges[0].directed:
        num *= 2
    return num / (n * (n - 1))


def avg_degree(network):
    m = network.number_of_edges()
    n = max(network.number_of_nodes(), 1)
    return m / n if network.edges[0].directed else 2 * m / n

# TODO: adapt betweenness_centrality and perhaps other statistics from
# https://github.com/networkdynamics/zenlib/blob/master/src/zen/algorithms/centrality.pyx

# TODO: Combo that lets the user set edge types to use (or all)

# Order of this list also defines execution priority
METHODS = (
    ("n", lambda network: network.number_of_nodes()),
    ("e", lambda network: network.number_of_edges()),
    ("degrees", lambda network: network.degrees(), "Degree", NODELEVEL, GENERAL),

    ("number_of_nodes", lambda n: n, "Number of nodes", GRAPHLEVEL, GENERAL),
    ("number_of_edges", lambda e: e, "Number of edges", GRAPHLEVEL, GENERAL),
    ("average_degree", lambda network: avg_degree(network), "Average degree", GRAPHLEVEL, GENERAL),
    ("density", lambda network: density(network), "Density", GRAPHLEVEL, GENERAL),

    ("shortest_paths", shortest_paths_nan_diag, "Shortest paths"),
    ("diameter",
     lambda shortest_paths: np.nanmax(shortest_paths), "Diameter", GRAPHLEVEL, GENERAL),
    ("radius",
     lambda shortest_paths: np.nanmin(np.nanmax(shortest_paths, axis=1)),
     "Radius", GRAPHLEVEL, GENERAL),
    ("average_shortest_path_length",
     lambda shortest_paths: np.nanmean(shortest_paths),
     "Average shortest path length", GRAPHLEVEL, GENERAL),

    ("number_strongly_connected_components", lambda network:
         csgraph.connected_components(network.edges[0].edges, False)[0],
         "Number of strongly connected components", GRAPHLEVEL, DIRECTED),
    ("number_weakly_connected_components", lambda network:
         csgraph.connected_components(network.edges[0].edges, True)[0],
         "Number of weakly connected components", GRAPHLEVEL, DIRECTED),

    ("in_degrees", lambda network: network.in_degrees(), "In-degree", NODELEVEL, GENERAL),
    ("out_degrees", lambda network: network.out_degrees(), "Out-degree", NODELEVEL, GENERAL),
    ("average_neighbour_degrees",
     lambda network, degrees: np.fromiter(
         (np.mean(degrees[network.neighbours(i)]) if degree else np.nan
          for i, degree in enumerate(degrees)),
         dtype=float, count=len(degrees)),
     "Average neighbor degree", NODELEVEL, GENERAL),
    ("degree_centrality",
     lambda degrees, n: n and degrees / (n - 1) if n > 1 else 0,
     "Degree centrality", NODELEVEL, GENERAL),
    ("in_degree_centrality",
     lambda in_degrees, n: in_degrees / (n - 1) if n > 1 else 0,
     "In-degree centrality", NODELEVEL, GENERAL),
    ("out_degree_centrality",
     lambda out_degrees, n: out_degrees / (n - 1) if n > 1 else 0,
     "Out-degree centrality", NODELEVEL, GENERAL),
    ("closeness_centrality",
     lambda shortest_paths: 1 / np.nanmean(shortest_paths, axis=1),
     "Closeness centrality", NODELEVEL, GENERAL)
)

MethodDefinition = namedtuple(
    "MethodDefinition",
    ["name", "func", "label", "level", "edge_constraint", "args"])


METHODS = {definition[0]:
    MethodDefinition(*(definition + ("", None, "", INTERNAL, GENERAL)[len(definition):]),
                     inspect.getfullargspec(definition[1]).args)
    for definition in METHODS}

"""
        ("degree_assortativity_coefficient", False, \
            "Degree assortativity coefficient", GRAPHLEVEL, \
                nx.degree_assortativity_coefficient if \
                hasattr(nx, "degree_assortativity_coefficient") else None),
        ("degree_pearson_correlation_coefficient", False, \
            "Degree pearson correlation coefficient", GRAPHLEVEL, \
            nx.degree_pearson_correlation_coefficient if\
            hasattr(nx, "degree_pearson_correlation_coefficient") else None),
        ("estrada_index", False, "Estrada index", GRAPHLEVEL, \
            nx.estrada_index if hasattr(nx, "estrada_index") else None),
        ("graph_clique_number", False, "Graph clique number", GRAPHLEVEL, nx.graph_clique_number),
        ("graph_number_of_cliques", False, "Graph number of cliques", GRAPHLEVEL, nx.graph_number_of_cliques),
        ("transitivity", False, "Graph transitivity", GRAPHLEVEL, nx.transitivity),
        ("average_clustering", False, "Average clustering coefficient", GRAPHLEVEL, nx.average_clustering),
        ("number_attracting_components", False, "Number of attracting components", GRAPHLEVEL, nx.number_attracting_components),

        ("clustering", False, "Clustering coefficient", NODELEVEL, nx.clustering),
        ("triangles", False, "Number of triangles", NODELEVEL, nx.triangles),
        ("square_clustering", False, "Squares clustering coefficient", NODELEVEL, nx.square_clustering),
        ("number_of_cliques", False, "Number of cliques", NODELEVEL, nx.number_of_cliques),
        ("betweenness_centrality", False, "Betweenness centrality", NODELEVEL, nx.betweenness_centrality),
        ("current_flow_closeness_centrality", False, "Information centrality", NODELEVEL, nx.current_flow_closeness_centrality),
        ("current_flow_betweenness_centrality", False, "Random-walk betweenness centrality", NODELEVEL, nx.current_flow_betweenness_centrality),
        ("approximate_current_flow_betweenness_centrality", False, \
            "Approx. random-walk betweenness centrality", NODELEVEL, \
            nx.approximate_current_flow_betweenness_centrality if \
            hasattr(nx, "approximate_current_flow_betweenness_centrality") \
                else None),
        ("eigenvector_centrality", False, "Eigenvector centrality", NODELEVEL, nx.eigenvector_centrality),
        ("eigenvector_centrality_numpy", False, "Eigenvector centrality (NumPy)", NODELEVEL, nx.eigenvector_centrality_numpy),
        ("load_centrality", False, "Load centrality", NODELEVEL, nx.load_centrality),
        ("core_number", False, "Core number", NODELEVEL, nx.core_number),
        ("eccentricity", False, "Eccentricity", NODELEVEL, nx.eccentricity),
        ("closeness_vitality", False, "Closeness vitality", NODELEVEL, nx.closeness_vitality),
        """


class WorkerThread(QThread):
    def __init__(self, method, data):
        super().__init__()
        self.method = method
        self.data = data

        self.stopped = 0
        self.result = None
        self.error = None
        self.is_terminated = False

    def run(self):
        args = tuple(self.data[arg] for arg in self.method.args)
        if any(arg is TERMINATED for arg in args):  # "in" doesn't work with np
            self.result = TERMINATED
        elif any(arg is ERRORED for arg in args):
            self.result = TERMINATED
        else:
            try:
                self.result = self.method.func(*args)
            except Exception as ex:
                self.error = ex
                print(ex)


class OWNxAnalysis(widget.OWWidget):
    name = 'Network Analysis'
    description = 'Statistical analysis of network data.'
    icon = 'icons/NetworkAnalysis.svg'
    priority = 6425

    resizing_enabled = False

    class Inputs:
        network = Input("Network", Network)
        items = Input("Items", Table)

    class Outputs:
        network = Output("Network", Network)
        items = Output("Items", Table)

    class Information(widget.OWWidget.Information):
        computing = Msg("Computing {}")

    want_main_area = False
    want_control_area = True

    auto_commit = Setting(False)
    enabled_methods = Setting(
        {"number_of_nodes", "number_of_edges", "average_degree"})

    def __init__(self):
        super().__init__()

        self.graph = None
        self.items = None          # items set by Items signal
        self.items_graph = None    # items set by graph.items by Network signal
        self.items_analysis = None # items to output and merge with analysis result

        self.known = {}
        self.running_jobs = {}
        # Indicates that node level statistics have changed or are pending to
        self._nodelevel_invalidated = False

        self.controlArea = QWidget(self.controlArea)
        self.layout().addWidget(self.controlArea)
        layout = QGridLayout()
        self.controlArea.setLayout(layout)
        layout.setContentsMargins(4, 4, 4, 4)
        tabs = gui.tabWidget(self.controlArea)
        tabs.setMinimumWidth(450)
        graph_indices = gui.createTabPage(tabs, "Graph-level indices",
                                          orientation=Qt.Horizontal)
        node_indices = gui.createTabPage(tabs, "Node-level indices",
                                         orientation=Qt.Horizontal)

        graph_methods = gui.vBox(graph_indices)
        gui.rubber(graph_indices)
        graph_labels = gui.vBox(graph_indices)

        node_methods = gui.vBox(node_indices)
        gui.rubber(node_indices)
        node_labels = gui.vBox(node_indices)
        graph_labels.layout().setAlignment(Qt.AlignRight)

        self.method_cbs = {}
        for method in METHODS.values():
            if method.level == INTERNAL:
                continue
            setattr(self, method.name, method.name in self.enabled_methods)
            setattr(self, "lbl_" + method.name, "")

            methods = node_methods if method.level == NODELEVEL else graph_methods
            labels = node_labels if method.level == NODELEVEL else graph_labels

            cb = gui.checkBox(
                methods, self, method.name, method.label,
                callback=lambda attr=method.name: self.method_clicked(attr)
            )
            self.method_cbs[method.name] = cb

            lbl = gui.label(labels, self, f"%(lbl_{method.name})s")
            labels.layout().setAlignment(lbl, Qt.AlignRight)
            setattr(self, "tool_" + method.name, lbl)
            # todo: is this accessible through controls?
        graph_indices.layout().addStretch(1)
        node_indices.layout().addStretch(1)

    @Inputs.network
    def set_graph(self, graph):
        self.cancel_job()
        self.graph = graph

        allowed_edge_types = {GENERAL, DIRECTED, UNDIRECTED}
        if graph is not None:
            allowed_edge_types.remove(UNDIRECTED if self.graph.edges[0].directed else DIRECTED)

        self.known = {}
        for name in METHODS:
            curr_method = METHODS[name]
            if curr_method.level == INTERNAL:
                continue

            lbl_obj = getattr(self, "tool_{}".format(name))
            cb_obj = self.method_cbs[name]

            # disable/re-enable valid graph indices
            if curr_method.edge_constraint not in allowed_edge_types:
                lbl_obj.setDisabled(True)
                cb_obj.setChecked(False)
                cb_obj.setEnabled(False)
            else:
                lbl_obj.setDisabled(False)
                cb_obj.setEnabled(True)

            setattr(self, f"lbl_{name}", "")

        if graph is not None:
            self.known["network"] = graph
            self.items_graph = graph.nodes

    @Inputs.items
    def set_items(self, items):
        self.items = items

    def handleNewSignals(self):
        if self.items is not None:
            self.items_analysis = self.items
        elif self.graph:
            self.items_analysis = self.graph.nodes
        else:
            self.items_analysis = None
        self._nodelevel_invalidated = True
        self.run_more_jobs()

    def needed_methods(self):
        # Preconditions for methods could be precomputed, so this function would
        # only compute the union of sets of conditions for enabled checkboxes
        tasks = [
            name for name in METHODS if getattr(self, name, False)]
        for name in tasks:
            tasks += [name for name in METHODS[name].args if name != "network"]
        tasks = set(tasks) - set(self.known)
        return [method for name, method in METHODS.items() if name in tasks]

    def run_more_jobs(self):
        known = set(self.known)
        needed = self.needed_methods()
        for method in needed:
            if method.name not in self.running_jobs:
                setattr(self, "lbl_" + method.name, "pending")
        doable = [method for method in needed
                  if method.name not in self.running_jobs
                  and set(method.args) <= known]
        free = max(1, QThread.idealThreadCount()) - len(self.running_jobs)
        if not doable:
            # This will output new data when everything is finished
            self.send_data()
        for method in doable[:free]:
            job = WorkerThread(method, self.known)
            job.finished.connect(lambda job=job: self.job_finished(job))
            self.running_jobs[method.name] = job
            job.start()
            if not method.level == INTERNAL:
                setattr(self, "lbl_" + method.name, "running")
        self.show_computing()

    def job_finished(self, job):
        method = job.method
        self.known[method.name] = job.result
        del self.running_jobs[method.name]
        self.set_label_for(method.name)
        self.run_more_jobs()

    def set_label_for(self, name):
        level = METHODS[name].level
        if level == INTERNAL:
            return
        value = self.known.get(name, None)
        txt = ""
        if getattr(self, name, False):
            if value is TERMINATED:
                txt = "terminated"
            elif value is ERRORED:
                txt = "error"
            elif value is None:
                txt = "computing" if name in self.running_jobs else "pending"
            elif level == GRAPHLEVEL:
                txt = f"{value:.4g}"
        setattr(self, "lbl_" + name, txt)

    def show_computing(self):
        computing = ", ".join(METHODS[name].label for name in self.running_jobs)
        self.Information.computing(computing, shown=bool(computing))

    def cancel_job(self, name=None):
        # This does not really work because functions called in those
        # threads do not observe the "is_terminated" flag and won't quit
        if name is None:
            to_stop = list(self.running_jobs)
        elif name in self.running_jobs:
            to_stop = [name]
        else:
            # This task is not running; but are its preconditions running?
            still_needed = self.needed_methods()
            to_stop = [
                name for name in (job.method.name for job in self.running_jobs)
                if METHODS[name] not in still_needed]
        for name in to_stop:
            job = self.running_jobs[name]
            job.is_terminated = True
            job.finished.disconnect()
            job.quit()
        for name in to_stop:
            job = self.running_jobs[name]
            job.wait()
            setattr(self, "lbl_" + name, "terminated")
            del self.running_jobs[name]
        self.show_computing()

    def onDeleteWidget(self):
        self.cancel_job()
        super().onDeleteWidget()

    def send_data(self):
        # Don't send when computation is still on, or it's done but no node
        # level statistics have changed
        if self.running_jobs or not self._nodelevel_invalidated:
            return
        self._nodelevel_invalidated = False
        if self.graph is None:
            self.Outputs.network.send(None)
            self.Outputs.items.send(None)
            return

        to_report = [
            method for attr, method in METHODS.items()
            if method.level == NODELEVEL
            and getattr(self, attr) and attr in self.known]
        items = self.items_analysis
        graph = self.graph
        n = graph.number_of_nodes()
        if isinstance(items, Table):
            dom = self.items_analysis.domain
            attrs, class_vars, metas = dom.attributes, dom.class_vars, dom.metas
            x, y, m = items.X, items.Y, items.metas
        else:
            attrs, class_vars, metas = [], [], []
            x = y = m = np.empty((n, 0))
        attrs += tuple(ContinuousVariable(method.label) for method in to_report)
        x = np.hstack(
            (x, ) + tuple(self.known[method.name].reshape((n, 1))
                          for method in to_report))
        domain = Domain(attrs, class_vars, metas)
        table = Table(domain, x, y, m)
        new_graph = Network(table, graph.edges, graph.name, graph.coordinates)
        self.Outputs.network.send(new_graph)
        self.Outputs.items.send(table)

    def method_clicked(self, name):
        if METHODS[name].level == NODELEVEL:
            self._nodelevel_invalidated = True
        if getattr(self, name):
            self.enabled_methods.add(name)
            if name in self.known:
                self.set_label_for(name)
                self.send_data()
            else:
                self.run_more_jobs()
        else:
            self.enabled_methods.remove(name)
            if name in self.running_jobs:
                self.cancel_job(name)
            else:
                self.set_label_for(name)
            self.send_data()

    def send_report(self):
        self.report_items("", items=[(method.label, f"{self.known[attr]:.4g}")
                                     for attr, method in METHODS.items()
                                     if method.level == GRAPHLEVEL
                                     and getattr(self, attr)
                                     and attr in self.known
                                     and isinstance(self.known[attr], (int, float))])


def main():
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.network.network.readwrite \
        import read_pajek, transform_data_to_orange_table
    from os.path import join, dirname

    #network = read_pajek(join(dirname(dirname(__file__)), 'networks', 'leu_by_genesets.net'))
    network = read_pajek(join(dirname(dirname(__file__)), 'networks', 'lastfm.net'))
    #network = read_pajek(join(dirname(dirname(__file__)), 'networks', 'Erdos02.net'))
    #transform_data_to_orange_table(network)
    WidgetPreview(OWNxAnalysis).run(set_graph=network)


if __name__ == "__main__":
    main()
