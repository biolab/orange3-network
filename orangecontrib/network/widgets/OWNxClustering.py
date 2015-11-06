from Orange.widgets import gui, widget, settings
from orangecontrib.network import Graph, community as cd


class OWNxClustering(widget.OWWidget):
    name = 'Network Clustering'
    description = 'Orange widget for community detection in networks.'
    icon = "icons/NetworkClustering.svg"
    priority = 6430

    inputs = [("Network", Graph, "setNetwork", widget.Default)]
    outputs = [("Network", Graph),
               ("Community Detection", cd.CommunityDetection)]

    resizing_enabled = False

    method = settings.Setting(0)
    iterations = settings.Setting(1000)
    hop_attenuation = settings.Setting(0.1)
    autoApply = settings.Setting(False)

    def __init__(self):
        super().__init__()

        self.net = None

        commit = lambda: self.commit()
        gui.spin(self.controlArea, self, "iterations", 1,
                   100000, 1, label="Max. iterations:",
                   callback=commit)
        ribg = gui.radioButtonsInBox(
            self.controlArea, self, "method",
            btnLabels=["Label propagation clustering (Raghavan et al., 2007)",
                    "Label propagation clustering (Leung et al., 2009)"],
            box="Clustering method", callback=commit)

        gui.doubleSpin(gui.indentedBox(ribg), self, "hop_attenuation",
                         0, 1, 0.01, label="Hop attenuation (delta): ")

        self.info = gui.widgetLabel(self.controlArea, ' ')

        gui.auto_commit(self.controlArea, self, "autoApply", 'Commit',
                        checkbox_label='Auto-commit')
        commit()

    def setNetwork(self, net):
        self.net = net
        self.commit()

    def commit(self):
        self.info.setText(' ')

        if self.method == 0:
            alg = cd.label_propagation
            kwargs = {'results2items': 1,
                      'iterations': self.iterations}

        elif self.method == 1:
            alg = cd.label_propagation_hop_attenuation
            kwargs = {'results2items': 1,
                      'iterations': self.iterations,
                      'delta': self.hop_attenuation}

        self.send("Community Detection", cd.CommunityDetection(alg, **kwargs))

        if self.net is None:
            self.send("Network", None)
            return

        labels = alg(self.net, **kwargs)

        self.info.setText('%d clusters found' % len(set(labels.values())))
        self.send("Network", self.net)


if __name__ == "__main__":
    from PyQt4.QtGui import *
    a = QApplication([])
    ow = OWNxClustering()
    ow.show()

    def setNetwork(signal, data, id=None):
        if signal == 'Network':
            ow.setNetwork(data)

    import OWNxFile
    from os.path import join, dirname
    owFile = OWNxFile.OWNxFile()
    owFile.send = setNetwork
    owFile.openFile(join(dirname(dirname(__file__)), 'networks', 'leu_by_genesets.net'))

    a.exec_()
    ow.saveSettings()
    owFile.saveSettings()
