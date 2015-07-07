import Orange.network
import Orange.network.community as cd
from Orange.widgets import gui, widget


class OWNxClustering(widget.OWWidget):
    name = 'Network Clustering'
    description = 'Orange widget for community detection in networks.'
    icon = "icons/NetworkClustering.svg"
    priority = 6430

    inputs = [("Network", Orange.network.Graph, "setNetwork", widget.Default)]
    outputs = [("Network", Orange.network.Graph),
               ("Community Detection", cd.CommunityDetection)]

    settingsList = ['method', 'iterationHistory', 'autoApply', 'iterations',
                    'hop_attenuation']

    def __init__(self):
        super().__init__()

        self.net = None
        self.method = 0
        self.iterationHistory = 0
        self.autoApply = 0
        self.iterations = 1000
        self.hop_attenuation = 0.1
        self.loadSettings()

        gui.spin(self.controlArea, self, "iterations", 1,
                   100000, 1, label="Iterations: ")
        ribg = gui.radioButtonsInBox(self.controlArea, self, "method",
                                       [], "Method", callback=self.cluster)
        gui.appendRadioButton(ribg, self, "method",
                        "Label propagation clustering (Raghavan et al., 2007)",
                        callback=self.cluster)

        gui.appendRadioButton(ribg, self, "method",
                        "Label propagation clustering (Leung et al., 2009)",
                        callback=self.cluster)
        gui.doubleSpin(gui.indentedBox(ribg), self, "hop_attenuation",
                         0, 1, 0.01, label="Hop attenuation (delta): ")

        self.info = gui.widgetLabel(self.controlArea, ' ')
        gui.checkBox(self.controlArea, self, "iterationHistory",
                       "Append clustering data on each iteration",
                       callback=self.cluster)
        gui.checkBox(self.controlArea, self, "autoApply",
                       "Commit automatically")
        gui.button(self.controlArea, self, "Commit",
                     callback=lambda b=True: self.cluster(b))

        self.cluster()

    def setNetwork(self, net):
        self.net = net
        if self.autoApply:
            self.cluster()

    def cluster(self, btn=False):
        if not btn and not self.autoApply:
            return

        self.info.setText(' ')

        if self.method == 0:
            alg = cd.label_propagation
            kwargs = {'results2items': 1,
                      'resultHistory2items': self.iterationHistory,
                      'iterations': self.iterations}

        elif self.method == 1:
            alg = cd.label_propagation_hop_attenuation
            kwargs = {'results2items': 1,
                      'resultHistory2items': self.iterationHistory,
                      'iterations': self.iterations,
                      'delta': self.hop_attenuation}

        self.send("Community Detection", cd.CommunityDetection(alg, **kwargs))

        if self.net is None:
            self.send("Network", None)
            return

        labels = alg(self.net, **kwargs)

        self.info.setText('%d clusters found' % len(set(labels.values())))
        self.send("Network", self.net)
