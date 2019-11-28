from AnyQt.QtCore import Qt, QThread
from Orange.data import Table
from Orange.widgets.widget import OWWidget
from gensim.models.callbacks import CallbackAny2Vec
from orangewidget import gui, settings, widget
from orangewidget.utils.signals import Input, Output

from orangecontrib.network import Network
from orangecontrib.network.network import embeddings

SPIN_WIDTH = 75
METHOD_NAMES = ["node2vec"]
NODE2VEC = 0


class EmbedderThread(QThread):
    def __init__(self, func):
        super().__init__()
        self.func = func
        self.result = None

    def __del__(self):
        self.quit()

    def run(self):
        self.result = self.func()


class ProgressBarUpdater(CallbackAny2Vec):
    def __init__(self, widget, num_epochs):
        self.widget = widget
        self.curr_epoch = 0
        self.num_epochs = num_epochs

    def on_epoch_begin(self, model):
        self.widget.progressBarSet(100 * (self.curr_epoch / self.num_epochs))

    def on_epoch_end(self, model):
        self.curr_epoch += 1


class OWNxEmbedding(OWWidget):
    name = "Network Embeddings"
    description = "Embed network elements"
    icon = "icons/NetworkFile.svg"  # TODO: a proper icon
    priority = 6450

    class Inputs:
        network = Input("Network", Network, default=True)

    class Outputs:
        items = Output("Items", Table)

    resizing_enabled = False
    want_main_area = False

    method = settings.Setting(NODE2VEC)

    p = settings.Setting(1.0)
    q = settings.Setting(1.0)
    walk_len = settings.Setting(50)
    num_walks = settings.Setting(10)
    emb_size = settings.Setting(128)
    window_size = settings.Setting(5)
    num_epochs = settings.Setting(1)

    auto_commit = settings.Setting(True)

    def __init__(self):
        super().__init__()
        self.network = None
        self.embedder = None
        self._worker_thread = None
        self._progress_updater = None

        def commit():
            return self.commit()

        gui.comboBox(self.controlArea, self, "method", label="Method: ",
                     items=[method for method in METHOD_NAMES], callback=commit)

        box = gui.widgetBox(self.controlArea, "Method parameters")
        gui.spin(box, self, "p", 0.0, 10.0, 0.1, label="Return parameter (p): ", spinType=float,
                 controlWidth=SPIN_WIDTH, callback=commit)
        gui.spin(box, self, "q", 0.0, 10.0, 0.1, label="In-out parameter (q): ", spinType=float,
                 controlWidth=SPIN_WIDTH, callback=commit)
        gui.spin(box, self, "walk_len", 1, 100_000, 1, label="Walk length: ",
                 controlWidth=SPIN_WIDTH, callback=commit)
        gui.spin(box, self, "num_walks", 1, 10_000, 1, label="Walks per node: ",
                 controlWidth=SPIN_WIDTH, callback=commit)
        gui.spin(box, self, "emb_size", 1, 10_000, 1, label="Embedding size: ",
                 controlWidth=SPIN_WIDTH, callback=commit)
        gui.spin(box, self, "window_size", 1, 20, 1, label="Context size: ",
                 controlWidth=SPIN_WIDTH, callback=commit)
        gui.spin(box, self, "num_epochs", 1, 100, 1, label="Number of epochs: ",
                 controlWidth=SPIN_WIDTH, callback=commit)

        self.info_label = gui.widgetLabel(self.controlArea, "")
        gui.auto_commit(self.controlArea, self, "auto_commit", "Commit",
                        checkbox_label="Auto-commit", orientation=Qt.Horizontal)
        commit()

    @Inputs.network
    def set_network(self, net):
        self.network = net
        self.commit()

    def commit(self):
        self.Warning.clear()
        self.info_label.setText("")

        # cancel existing computation if running
        if self._worker_thread is not None:
            self._worker_thread.finished.disconnect()
            self._worker_thread.quit()
            self._worker_thread = None

        if self.network is None:
            self.Outputs.items.send(None)
            return

        self._progress_updater = ProgressBarUpdater(self, self.num_epochs)
        if self.method == NODE2VEC:
            self.embedder = embeddings.Node2Vec(self.p, self.q, self.walk_len, self.num_walks,
                                                self.emb_size, self.window_size, self.num_epochs,
                                                callbacks=[self._progress_updater])

        self._worker_thread = EmbedderThread(lambda: self.embedder(self.network))
        self._worker_thread.finished.connect(self.on_finished)
        self.progressBarInit()
        self.progressBarSet(1e-5)
        self._worker_thread.start()

    def on_finished(self):
        output = self._worker_thread.result
        self._worker_thread = None
        self._progress_updater = None
        self.progressBarFinished()
        self.Outputs.items.send(output)

    def onDeleteWidget(self):
        if self._worker_thread is not None:
            self._worker_thread.finished.disconnect()
            self._worker_thread.quit()
        super().onDeleteWidget()


if __name__ == "__main__":
    from AnyQt.QtWidgets import QApplication
    a = QApplication([])
    ow = OWNxEmbedding()
    ow.show()

    def set_network(data, id=None):
        ow.set_network(data)

    import OWNxFile
    from os.path import join, dirname
    owFile = OWNxFile.OWNxFile()
    owFile.Outputs.network.send = set_network
    owFile.open_net_file(join(dirname(dirname(__file__)), "networks", "davis.net"))

    a.exec_()
    ow.saveSettings()
    owFile.saveSettings()

