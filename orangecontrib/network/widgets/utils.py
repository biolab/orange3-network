import numpy as np
from Orange.data import Table, StringVariable, Domain


def items_from_distmatrix(matrix):
    if matrix.row_items is not None:
        row_items = matrix.row_items
        if isinstance(row_items, Table):
            if matrix.axis == 1:
                items = row_items
            else:
                items = [[v.name] for v in row_items.domain.attributes]
        else:
            items = [[str(x)] for x in matrix.row_items]
    else:
        items = [[str(i)] for i in range(1, 1 + matrix.shape[0])]
    if not isinstance(items, Table):
        items = Table.from_list(
            Domain([], metas=[StringVariable('label')]),
            items)
    return items


def weights_from_distances(weights):
    weights = weights.astype(np.float64)

    if weights.size == 0:
        return weights

    mi, ma = np.nanmin(weights), np.nanmax(weights)
    assert mi >= 0, "All distances must be positive"

    if ma - mi < 1e-6:
        return np.ones(weights.shape)

    weights /= ma
    weights *= np.log(10)
    np.exp(weights, out=weights)
    np.reciprocal(weights, out=weights)

    return weights
