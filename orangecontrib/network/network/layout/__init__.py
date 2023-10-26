import time

import numpy as np

from ._fr_layout import fruchterman_reingold_step

TEST_ITERATIONS = 10


def fruchterman_reingold(pos,
                         edges, weighted: bool,
                         allowed_time,
                         k,
                         init_temp=0.05,
                         sample_ratio=None,
                         callback_interval=0.5, callback=None):

    def run_iterations(n_iterations, temperatures, callback=callback):
        last_call = time.perf_counter()
        for iteration in range(n_iterations):
            if sample_ratio is not None:
                np.random.shuffle(sample)
            fruchterman_reingold_step(pos, sample[:sample_size],
                                      edge_src, edge_dst, edge_weights,
                                      k, temperatures[iteration], disp)

            if callback and \
                    (now := time.perf_counter()) - last_call > callback_interval:
                progress = iteration / n_iterations * 100
                callback(pos, progress)
                last_call = now

    n_nodes = len(pos)

    edge_weights = edges.data if weighted else np.empty(0)
    edge_src, edge_dst = edges.row.astype(np.int32), edges.col.astype(np.int32)

    disp = np.empty((n_nodes, 2))

    if sample_ratio is not None:
        sample_size = max(int(round(n_nodes * sample_ratio)), 1)
        sample = np.arange(n_nodes, dtype=np.int32)
    else:
        sample = np.zeros(0, dtype=np.int32)
        sample_size = 0

    temperatures = np.linspace(init_temp, 0.01, TEST_ITERATIONS)
    start_time = time.perf_counter()
    run_iterations(TEST_ITERATIONS, temperatures, callback=None)
    elapsed_time = time.perf_counter() - start_time

    iterations = int(allowed_time / (elapsed_time / TEST_ITERATIONS))
    if TEST_ITERATIONS > iterations:
        return
    iterations = min(iterations, 100 * n_nodes)
    temperatures = np.linspace(0.1, 0.015, iterations) ** 2
    run_iterations(iterations, temperatures)
