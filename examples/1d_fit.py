# -*- coding: utf-8 -*-
"""
1-D fit — Akima's benchmark dataset.

The classic interpolation stress test: a long flat run followed by a
steep rise, where polynomial and spline methods tend to overshoot.
EBF is massive overkill in 1-D, but this is the smallest possible
end-to-end workflow: fit, predict on a dense grid, plot.

Also shows the combined-array input convenience — ``fit()`` accepts a
single (n_points, n_dims+1) array whose last column is the output.
"""
import numpy as np
import matplotlib.pyplot as plt

import ebf

if __name__ == "__main__":

    # --- Akima sample points (last column is the output) ---
    x = np.array([0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15], dtype=float)
    y = np.array([10, 10, 10, 10, 10, 10, 10.5, 15, 50, 60, 85])
    data = np.column_stack([x, y])

    # --- Train ---
    model = ebf.EBF(n_nodes=11)
    model.fit(
        data,               # combined array — last column is y
        steps=20000,
        var_weight=0.5,     # strong node-spread regularization for sparse 1-D data
        seed=42,
    )
    print("Node positions:\n", model.get_nodes())

    # --- Diagnostics ---
    ebf.convergence_plot(model)
    ebf.correlation_plot(y, model.predict(x.reshape(-1, 1)))

    # --- Fitted curve on a dense grid ---
    query = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
    y_fit = model.predict(query)

    fig, ax = plt.subplots()
    ax.plot(query, y_fit, label='EBF fit')
    ax.scatter(x, y, color='black', label='Data', zorder=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.set_title('Akima benchmark — 1-D EBF fit')

    plt.show()
