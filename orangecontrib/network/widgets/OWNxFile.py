from os import path
from itertools import chain, product

from PyQt4.QtGui import *
from PyQt4.QtCore import *

import Orange
from Orange.widgets import gui, widget, settings
import orangecontrib.network as network


NONE = "(none)"


class OWNxFile(widget.OWWidget):
    name = "Network File"
    description = "Read network graph file in Pajek or GML format."
    icon = "icons/NetworkFile.svg"
    priority = 6410

    outputs = [("Network", network.Graph),
               ("Items", Orange.data.Table)]

    resizing_enabled = False

    recentFiles = settings.Setting([])
    recentDataFiles = settings.Setting([])
    auto_table = settings.Setting(True)

    def __init__(self):
        super().__init__()

        self.domain = None
        self.graph = None
        self.auto_items = None

        self.net_index = 0
        self.data_index = 0

        #GUI
        self.controlArea.layout().setMargin(4)
        self.box = gui.widgetBox(self.controlArea, box="Graph File", orientation="vertical")
        hb = gui.widgetBox(self.box, orientation="horizontal")
        self.filecombo = gui.comboBox(hb, self, "net_index", callback=self.selectNetFile)
        self.filecombo.setMinimumWidth(250)
        button = gui.button(hb, self, '...', callback=self.browseNetFile, disabled=0)
        button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        button = gui.button(hb, self, 'Reload', callback=self.reload)
        button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        gui.checkBox(self.box, self, "auto_table", "Build graph data table automatically",
                     callback=self.selectNetFile)

        self.databox = gui.widgetBox(self.controlArea, box="Vertices Data File", orientation="horizontal")
        self.datacombo = gui.comboBox(self.databox, self, "data_index", callback=self.selectDataFile)
        self.datacombo.setMinimumWidth(250)
        button = gui.button(self.databox, self, '...', callback=self.browseDataFile, disabled=0)
        button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        button = gui.button(self.databox, self, 'Reload', callback=self.reload_data)
        button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))

        # info
        box = gui.widgetBox(self.controlArea, "Info")
        self.info = gui.widgetLabel(box, 'No data loaded.')

        gui.rubber(self.controlArea)
        self.resize(150, 100)

        self.populate_comboboxes()
        self.reload()

    def reload(self):
        if self.recentFiles:
            self.selectNetFile()

    def reload_data(self):
        if self.recentDataFiles:
            self.selectDataFile()

    def populate_comboboxes(self):
        self.filecombo.clear()
        for file in self.recentFiles or (NONE,):
            self.filecombo.addItem(path.basename(file))
        self.filecombo.addItem("Browse documentation networks...")
        self.filecombo.updateGeometry()

        self.datacombo.clear()
        for file in self.recentDataFiles:
            self.datacombo.addItem(path.basename(file))
        self.datacombo.addItem(NONE)
        self.datacombo.updateGeometry()

    def selectNetFile(self):
        """user selected a graph file from the combo box"""
        if self.net_index > len(self.recentFiles) - 1:
            self.browseNetFile(True)
        elif self.net_index:
            self.recentFiles.insert(0, self.recentFiles.pop(self.net_index))
            self.net_index = 0
            self.populate_comboboxes()
        if self.recentFiles:
            self.openNetFile(self.recentFiles[0])

    def selectDataFile(self):
        if self.data_index > len(self.recentDataFiles) - 1:
            return self.openDataFile(NONE)
        self.recentDataFiles.insert(0, self.recentDataFiles.pop(self.data_index))
        self.data_index = 0
        self.populate_comboboxes()
        self.openDataFile(self.recentDataFiles[0])

    def readingFailed(self, message=''):
        self.graph = None
        self.send("Network", None)
        self.send("Items", None)
        self.info.setText('No data loaded.\n' + message)

    def openNetFile(self, filename):
        """Read network from file."""
        if path.splitext(filename)[1].lower() not in network.readwrite.SUPPORTED_READ_EXTENSIONS:
            return self.readingFailed('Network file type not supported')

        G = network.readwrite.read(filename, auto_table=self.auto_table)
        if G is None:
            return self.readingFailed('Error reading file "{}"'.format(filename))

        info = (('Directed' if G.is_directed() else 'Undirected') + ' graph',
                '{} nodes, {} edges'.format(G.number_of_nodes(), G.number_of_edges()),
                'Vertices data generated from graph' if self.auto_table else '')
        self.info.setText('\n'.join(info))

        self.auto_items = G.items()
        assert self.auto_table or self.auto_items is None, \
            (self.auto_table, self.auto_items)

        self.graph = G
        self.warning(0)

        # Find items data file for selected network
        for basename, ext in product((filename,
                                      path.splitext(filename)[0],
                                      path.splitext(filename)[0] + '_items'),
                                     ('.tab', '.tsv', '.csv')):
            candidate = basename + ext
            if path.exists(candidate):
                try: self.recentDataFiles.remove(candidate)
                except ValueError: pass
                self.recentDataFiles.insert(0, candidate)
                self.data_index = 0
                self.populate_comboboxes()
                self.openDataFile(self.recentDataFiles[0])
                break
        else:
            self.data_index = len(self.recentDataFiles)

        self.send("Network", G)
        self.send("Items", G.items() if G else None)

    def openDataFile(self, filename):
        self.error(1)
        self.warning(0)
        if filename == NONE:
            if self.graph:
                self.graph.set_items(self.auto_items)
                self.info.setText(self.info.text().rpartition('\n')[0] + '\n' +
                                  ('Vertices data generated from graph'
                                   if self.auto_items is not None else
                                   "No vertices data file specified"))
        else:
            self.readDataFile(filename)
        self.send("Network", self.graph)
        self.send("Items", self.graph.items() if self.graph else None)

    def readDataFile(self, filename):
        if not self.graph:
            self.warning(0, 'No network file loaded. Load the network first.')
            return

        table = Orange.data.Table.from_file(filename)

        if len(table) != self.graph.number_of_nodes():
            self.error(1, "Vertices data length does not match the number of vertices")
            self.populate_comboboxes()
            return

        items = self.auto_items
        if items is not None and len(items) == len(table):
            domain = [v for v in chain(items.domain.attributes,
                                       items.domain.class_vars,
                                       items.domain.metas)
                      if v.name not in table.domain]
            if domain:
                table = Orange.data.Table.concatenate([table, items[:, domain]])

        self.graph.set_items(table)
        self.info.setText(self.info.text().rpartition('\n')[0] + '\n' +
                          'Vertices data added')

    def browseNetFile(self, browse_demos=False):
        """user pressed the '...' button to manually select a file to load"""
        if browse_demos:
            from pkg_resources import load_entry_point
            startfile = next(load_entry_point("Orange3-Network",
                                              "orange.data.io.search_paths",
                                              "network")())[1]
        else:
            startfile = self.recentFiles[0] if self.recentFiles else '.'

        filename = QFileDialog.getOpenFileName(
            self, 'Open a Network File', startfile,
            ';;'.join(("All network files (*{})".format(
                           ' *'.join(network.readwrite.SUPPORTED_READ_EXTENSIONS)),
                       "NetworkX graph as Python pickle (*.gpickle)",
                       "NetworkX edge list (*.edgelist)",
                       "Pajek files (*.net *.pajek)",
                       "GML files (*.gml)",
                       "All files (*)")))

        if not filename: return
        try: self.recentFiles.remove(filename)
        except ValueError: pass
        self.recentFiles.insert(0, filename)

        self.populate_comboboxes()
        self.net_index = 0
        self.selectNetFile()

    def browseDataFile(self):
        if self.graph is None:
            self.warning(0, 'No network file loaded. Load the network first.')
            return

        startfile = (self.recentDataFiles[0] if self.recentDataFiles else
                     path.dirname(self.recentFiles[0]) if self.recentFiles else
                     '.')

        filename = QFileDialog.getOpenFileName(
            self, 'Open a Vertices Data File', startfile,
            'Data files (*.tab *.tsv *.csv);;'
            'All files(*)')

        if not filename: return
        try: self.recentDataFiles.remove(filename)
        except ValueError: pass
        self.recentDataFiles.insert(0, filename)
        self.populate_comboboxes()
        self.data_index = 0
        self.selectDataFile()

    def sendReport(self):
        self.reportSettings("Network file",
                            [("File name", self.filecombo.currentText()),
                             ("Vertices", self.graph.number_of_nodes()),
                             hasattr(self.graph, "is_directed") and ("Directed", gui.YesNo[self.graph.is_directed()])])
        self.reportSettings("Vertices meta data", [("File name", self.datacombo.currentText())])
        self.reportData(self.graph.items(), None)
        self.reportData(self.graph.links(), None, None)

if __name__ == "__main__":
    from PyQt4.QtGui import QApplication
    a = QApplication([])
    owf = OWNxFile()
    owf.show()
    a.exec_()
    owf.saveSettings()
