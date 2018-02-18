import networkx as nx
import numpy as np

from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.network.widgets.OWNxClustering import OWNxClustering
from orangecontrib.network.network import Graph


class TestOWNxClustering(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(
            OWNxClustering, stored_settings={'autoApply': False},
        )  # type: OWNxClustering

    def test_does_not_crash_when_cluster_variable_already_exists(self):
        # Prepare some dummy data
        x, clusters = np.ones((5, 1)), np.ones((5, 1))

        data_var = ContinuousVariable('Data')
        cluster_var = DiscreteVariable('Cluster', ('C1', 'C2'))
        data = Table.from_numpy(
            Domain([data_var], metas=[cluster_var]),
            X=x, metas=clusters,
        )
        graph = Graph(nx.complete_graph(5), '5-Clique')
        graph.set_items(data)

        # Should not crash
        self.send_signal(self.widget.Inputs.network, graph)
        self.widget.unconditional_commit()

        # There should be an additional cluster column
        output = self.get_output(self.widget.Outputs.network).items()
        self.assertEqual(len(data.domain.metas) + 1, len(output.domain.metas))
