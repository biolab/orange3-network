try:
    from gensim.models.callbacks import CallbackAny2Vec
except ImportError:
    CallbackAny2Vec = None

from AnyQt.QtCore import Qt, QThread, pyqtSignal as Signal, QObject
from AnyQt.QtWidgets import QFormLayout, QLabel

from orangewidget import gui, settings
from orangewidget.utils.signals import Input, Output
from orangewidget.utils.widgetpreview import WidgetPreview
from orangewidget.widget import Msg
from Orange.data import Table
from Orange.widgets.widget import OWWidget

from orangecontrib.network import Network
from orangecontrib.network.network import readwrite
if CallbackAny2Vec is not None:
    from orangecontrib.network.network import embeddings


class EmbedderThread(QThread):
    def __init__(self, func):
        super().__init__()
        self.func = func
        self.result = None

    def __del__(self):
        self.quit()

    def run(self):
        self.result = self.func()


class ProgressBarUpdater(CallbackAny2Vec, QObject):
    progress_changed = Signal(float)

    def __init__(self, widget, num_epochs):
        super().__init__()
        self.widget = widget
        self.curr_epoch = 0
        self.num_epochs = num_epochs

    def on_epoch_begin(self, model):
        if self.widget is None:
            return

        self.progress_changed.emit(100 * (self.curr_epoch / self.num_epochs))

    def on_epoch_end(self, model):
        self.curr_epoch += 1


class OWNxEmbedding(OWWidget):
    name = "Network Embeddings"
    description = "Embed network elements"
    icon = "icons/NetworkEmbedding.svg"
    priority = 6450

    class Inputs:
        network = Input("Network", Network, default=True)

    class Outputs:
        items = Output("Embeddings", Table)

    class Error(OWWidget.Error):
        unsupported_gensim = Msg(
            "This widget requires gensim, which is unsupported in Python>=3.14.")

    resizing_enabled = False
    want_main_area = False

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

        _labels = []
        def spin(label, var, min_, max_, step):
            label = QLabel(label + ":")
            _labels.append(label)
            spin = gui.spin(
                None, self, var, min_, max_, step,
                spinType=float if isinstance(step, float) else int,
                controlWidth=75, alignment=Qt.AlignRight,
                callback=self.commit.deferred)
            layout.addRow(label, spin)

        layout = QFormLayout()
        gui.widgetBox(self.controlArea, box="Random Walk", orientation=layout)
        spin("Return parameter", "p", 0.0, 10.0, 0.1)
        spin("In-out parameter", "q", 0.0, 10.0, 0.1)
        spin("Walk length", "walk_len", 1, 100_000, 1)
        spin("Walks per node", "num_walks", 1, 10_000, 1)

        layout = QFormLayout()
        gui.widgetBox(self.controlArea, box="Embedding", orientation=layout)
        spin("Embedding dimension", "emb_size", 1, 10_000, 1)
        spin("Window size", "window_size", 1, 20, 1)
        spin("Number of epochs", "num_epochs", 1, 100, 1)

        width = max(label.sizeHint().width() for label in _labels)
        for label in _labels:
            label.setMinimumWidth(width)

        gui.auto_commit(self.controlArea, self, "auto_commit", "Apply")

        if CallbackAny2Vec is None:
            self.Error.unsupported_gensim()
            self.controlArea.setDisabled(True)

    @Inputs.network
    def set_network(self, net):
        self.network = net
        self.commit.now()

    @gui.deferred
    def commit(self):
        if CallbackAny2Vec is None:
            return
        self.Warning.clear()

        # cancel existing computation if running
        if self._worker_thread is not None:
            self._worker_thread.finished.disconnect()
            self._worker_thread.quit()
            self._worker_thread = None

        if self.network is None:
            self.Outputs.items.send(None)
            return

        self._progress_updater = ProgressBarUpdater(self, self.num_epochs)
        self._progress_updater.progress_changed.connect(self.progressBarSet)
        self.embedder = embeddings.Node2Vec(self.p, self.q, self.walk_len, self.num_walks,
                                            self.emb_size, self.window_size, self.num_epochs,
                                            callbacks=[self._progress_updater])

        self._worker_thread = EmbedderThread(lambda: self.embedder(self.network))
        self._worker_thread.finished.connect(self.on_finished)
        self.progressBarInit()
        self._worker_thread.start()

    def on_finished(self):
        output = self._worker_thread.result
        self._worker_thread = None
        self._progress_updater = None
        self.progressBarFinished()
        self.Outputs.items.send(output)

    def onDeleteWidget(self):
        if self._worker_thread is not None:
            # prevent the callback from trying to access deleted widget object
            self._progress_updater.widget = None
            self._worker_thread.finished.disconnect()
            self._worker_thread.quit()
        super().onDeleteWidget()


def main():
    from os.path import join, dirname

    davis = join(dirname(__file__), "..", "networks", "davis.net")
    network = readwrite.read_pajek(davis)
    WidgetPreview(OWNxEmbedding).run(network)

if __name__ == "__main__":
    main()