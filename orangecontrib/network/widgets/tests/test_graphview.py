import unittest
from unittest.mock import Mock

import numpy as np
from AnyQt.QtCore import QLineF
from AnyQt.QtGui import QPen, QBrush, QColor

from Orange.widgets.tests.base import GuiTest
from ..graphview import PlotVarWidthCurveItem

class TestPlotVarWidthCurveItem(GuiTest):
    def line_to_tuple(self, line: QLineF):
        return line.x1(), line.y1(), line.x2(), line.y2()

    def test_coordinates(self):
        curve = PlotVarWidthCurveItem(
            directed=False,
            x=np.arange(4, dtype=float), y=np.arange(10, 14, dtype=float),
            size=np.zeros(4))
        painter = Mock()
        painter.worldTransform = Mock()
        painter.worldTransform.return_value.m11 = Mock(return_value=1)
        painter.worldTransform.return_value.m22 = Mock(return_value=1)

        curve.paint(painter, None, None)
        self.assertEqual(
            [self.line_to_tuple(call[0][0])
             for call in painter.drawLine.call_args_list],
            [(0, 10, 1, 11), (2, 12, 3, 13)])

        painter.reset_mock()
        curve.setData(
            x=np.arange(2, dtype=float), y=np.arange(10, 12, dtype=float),
            size=np.zeros(2))
        curve.paint(painter, None, None)
        self.assertEqual(
            [self.line_to_tuple(call[0][0])
             for call in painter.drawLine.call_args_list],
            [(0, 10, 1, 11)])

        painter.reset_mock()
        curve.setData(
            np.arange(100, dtype=float), np.arange(100, 200, dtype=float),
            size=np.zeros(100))
        curve.paint(painter, None, None)
        self.assertEqual(
            [self.line_to_tuple(call[0][0])
             for call in painter.drawLine.call_args_list],
            [(2 * x, 2 * x + 100, 2 * x + 1, 2 * x + 101) for x in range(50)])

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
            widths=np.arange(1000, 1100), size=np.ones(100))
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
