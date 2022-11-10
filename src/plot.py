import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def display(qtable,num_rows,num_cols):
    Q = [[np.max(qtable[x, y, :]) for x in range(1, num_rows - 1)] for y in
         range(1, num_cols - 1)]
    Q = np.reshape(Q, (num_rows - 2, num_cols - 2))
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    plot_values(ax1, Q.T)
    plt.show()


def plot_values(ax: Axes, V: np.ndarray):
    # reshape the state-value function
    # plot the state-value function
    im = ax.imshow(V, cmap='cool')
    for (j, i), label in np.ndenumerate(V):
        ax.text(i, j, np.round(label, 1), ha='center', va='center', fontsize=5)
    ax.tick_params(bottom='off', left='off', labelbottom='off', labelleft='off')