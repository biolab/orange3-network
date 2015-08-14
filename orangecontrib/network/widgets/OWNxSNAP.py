import sys
import os.path
import itertools

from PyQt4.QtGui import *

import Orange.data
import orangecontrib.network as network
from Orange.widgets import gui, widget, settings


class OWNxSNAP(widget.OWWidget):
    name = "SNAP"
    description = "Read networks from Stanford Large Network Dataset Collection."
    icon = "icons/SNAP.svg"
    priority = 6415

    outputs = [("Network", network.Graph),
               ("Items", Orange.data.Table)]

    last_total = settings.Setting(24763)

    def __init__(self):
        super().__init__()

        self.last_total = 24763

        self.networks = []
        self.tables = []

        image = QPixmap(os.path.dirname(__file__) + "/icons/snap_logo.png")

        imageLabel = QLabel(self)
        imageLabel.setPixmap(image)
        self.controlArea.layout().addWidget(imageLabel)

        self.controlArea.layout().addStretch(1)

        lbl = QLabel("<a href='http://snap.stanford.edu/data/'>http://snap.stanford.edu/data</a>", self)
        lbl.setOpenExternalLinks(True)
        self.controlArea.layout().addWidget(lbl)

        scrollArea = QScrollArea(self.mainArea)
        self.mainArea.layout().addWidget(scrollArea)

        self.network_list = gui.widgetBox(self.mainArea, addToLayout=False)
        self.network_list.layout().setSizeConstraint(QLayout.SetFixedSize);
        scrollArea.setWidget(self.network_list);

        self.snap = network.snap.SNAP()
        self.snap.get_network_list(self.add_tables, self.progress_callback)
        self.progressBarInit()
        self.setMinimumSize(960, 600)

    def add_tables(self, networks):
        from PyQt4.QtCore import SIGNAL
        self.networks = networks
        self.tables = []

        if networks is None:
            return

        networks.sort(key=lambda net: net.repository)
        for k,g in itertools.groupby(networks, key=lambda net: net.repository):
            network_group = list(g)

            if len(network_group) > 0:
                self.network_list.layout().addWidget(QLabel("<h3>" + network_group[0].repository + "</h3>"))
                table = gui.table(self.network_list, rows=len(network_group), columns=5, selectionMode = -1, addToLayout = 1)
                table.setHorizontalHeaderLabels(['Name', 'Type', 'Nodes', 'Edges', 'Description'])
                f = table.font()
                f.setPointSize(9)
                table.setFont(f)
                table.verticalHeader().hide()
                table.setSelectionMode(QAbstractItemView.SingleSelection)
                table.setSelectionBehavior(QAbstractItemView.SelectRows)
                self.connect(table, SIGNAL('itemSelectionChanged()'), lambda table=table: self.select_network(table))

                for i, net in enumerate(network_group):
                    lbl = QLabel("<a href='"+ net.link +"'>" + net.name + "</a>")
                    lbl.setOpenExternalLinks(True)
                    table.setCellWidget(i, 0, lbl)
                    gui.tableItem(table, i, 1, net.type)
                    gui.tableItem(table, i, 2, net.nodes)
                    gui.tableItem(table, i, 3, net.edges)
                    gui.tableItem(table, i, 4, net.description)

                table.setFixedSize(712, 100)
                table.setColumnWidth(0, 120)
                table.setColumnWidth(1, 80)
                table.setColumnWidth(2, 80)
                table.setColumnWidth(3, 80)
                table.setColumnWidth(4, 350)
                table.resizeRowsToContents()
                table.setFixedSize(712, sum(table.rowHeight(i) for i in range(len(networks))) + 27)
                self.tables.append(table)

                gui.separator(self.network_list, 10, 10)

    def download_progress(self, numblocks, blocksize, filesize):
        try:
            percent = min((numblocks*blocksize*100)/filesize, 100)
            self.progressBarSet(percent)
        except:
            percent = 100
            if numblocks != 0:
                print(str(percent)+'%')

    def select_network(self, selected_table):
        for table in self.tables:
            selected = table.selectedIndexes()
            if len(selected) > 0:
                row = selected[0].row()
                fn = selected_table.cellWidget(row, 0).text()
                network_info = self.snap.get_network(fn[fn.index('>')+1:-4])
                self.progressBarInit()
                network = network_info.read(progress_callback=self.download_progress)
                self.progressBarFinished()
                self.send('Network', network)

            if table is not selected_table:
                table.clearSelection()

    def progress_callback(self, done, total):
        if done == total:
            self.progressBarFinished()
            return

        if total > 0:
            self.progressBarSet(int(done/total * 100))
        else:
            self.progressBarSet(int(done * 100 / self.last_total))


if __name__ == "__main__":
    from PyQt4.QtGui import QApplication
    a=QApplication(sys.argv)
    owf=OWNxSNAP()
    owf.show()
    a.exec_()
    owf.saveSettings()
