import unittest
from unittest.mock import Mock

import numpy as np
from AnyQt.QtCore import QLineF
from AnyQt.QtGui import QPen, QBrush, QColor

from Orange.widgets.tests.base import GuiTest
from orangecontrib.network.widgets.graphview import PlotVarWidthCurveItem

class TestPlotVarWidthCurveItem(GuiTest):
    def line_to_tuple(self, line: QLineF):
        return line.x1(), line.y1(), line.x2(), line.y2()

    def test_coordinates(self):
        curve = PlotVarWidthCurveItem(
            directed=False,
            x=np.array([0, 18, 2, 2]), y=np.array([10, 10, 20, 40]),
            size=np.ones(4))
        painter = Mock()
        painter.worldTransform = Mock()
        painter.worldTransform.return_value.m11 = Mock(return_value=1)
        painter.worldTransform.return_value.m22 = Mock(return_value=1)

        curve.pen.width = lambda: 3
        curve.paint(painter, None, None)
        # coordinates are moved by sizes + width / 3 = 1 + 1 = 5
        self.assertEqual(
            [self.line_to_tuple(call[0][0])
             for call in painter.drawLine.call_args_list],
            [(2, 10, 16, 10), (2, 22, 2, 38)])

        painter.reset_mock()
        curve.pen.width = lambda: 6
        curve.setData(
            x=np.array([2, 2]), y=np.array([3, 20]),
            size=np.ones(2))
        curve.paint(painter, None, None)
        self.assertEqual(
            [self.line_to_tuple(call[0][0])
             for call in painter.drawLine.call_args_list],
            [(2, 6, 2, 17)])

        painter.reset_mock()
        s = np.arange(100) / 120
        curve.setData(
            np.arange(100, dtype=float), np.arange(100, 200, dtype=float),
            size=s)
        curve.paint(painter, None, None)
        s2 = np.sqrt(2) / 2
        exp = np.array([self.line_to_tuple(call[0][0])
                        for call in painter.drawLine.call_args_list])
        act = np.array([(2 * x + (2 + s[2 * x]) * s2,
                         2 * x + 100 + (2 + s[2 * x]) * s2,
                         2 * x + 1 - (2 + s[2 * x + 1]) * s2,
                         2 * x + 101 - (2 + s[2 * x + 1]) * s2) for x in range(50)])
        np.testing.assert_almost_equal(exp, act)

        painter.reset_mock()
        curve.setData(None, None)
        curve.paint(painter, None, None)
        painter.drawline.assert_not_called()

    def test_widths(self, ):
        def draw_line(*_):
            pens.append(painter.setPen.call_args[0][0].width())

        painter = Mock()
        painter.drawLine = draw_line
        painter.worldTransform = Mock()
        painter.worldTransform.return_value.m11 = Mock(return_value=1)
        painter.worldTransform.return_value.m22 = Mock(return_value=1)

        curve = PlotVarWidthCurveItem(
            directed=False,
            x=np.arange(4, dtype=float), y=np.arange(10, 14, dtype=float),
            size=np.ones(4), widths=np.array([2, 3]))

        pens = []
        curve.paint(painter, None, None)
        self.assertEqual(pens, [2, 3])

        curve.setWidths(np.array([10, 11]))
        pens = []
        curve.paint(painter, None, None)
        self.assertEqual(pens, [10, 11])

        curve.setData(
            np.arange(100, dtype=float), np.arange(100, 200, dtype=float),
            widths=np.arange(1000, 1050), size=np.ones(100))
        pens = []
        curve.paint(painter, None, None)
        self.assertEqual(pens, list(range(1000, 1050)))

        curve.setWidths(None)
        curve.setPen(QPen(QBrush(QColor(0, 0, 0)), 42))
        pens = []
        curve.paint(painter, None, None)
        self.assertEqual(pens, [42] * 50)

        curve.setData(
            np.arange(10, dtype=float), np.arange(100, 110, dtype=float),
            pen=QPen(QBrush(QColor(0, 0, 0)), 10), size=np.ones(10))
        pens = []
        curve.paint(painter, None, None)
        self.assertEqual(pens, [10] * 5)


if __name__ == "__main__":
    unittest.main()
