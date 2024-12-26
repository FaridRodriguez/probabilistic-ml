import numpy as np
import matplotlib.pyplot as plt

from src.common import eda_utils


def bootstrap_replicate(data, func):
    """Generate bootstrap replicate of 1-dimensional data."""
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)


def draw_bootsrap_replicates(data, func, size=1):
    """Draw bootstrap replicates."""
    bs_replicates = np.empty(size)
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate(data, func)
    return bs_replicates


def draw_bootrap_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""
    inds = np.arange(start=0, stop=len(x), step=1)
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(x=bs_x, y=bs_y, deg=1)
    return bs_slope_reps, bs_intercept_reps


def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""
    data = np.concatenate((data1, data2))
    permuted_data = np.random.permutation(data)
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]
    return perm_sample_1, perm_sample_2


def draw_permutation_replicates(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""
    perm_replicates = np.empty(size)
    for i in range(size):
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)
    return perm_replicates


def plot_permutation_ecdfs(data1, data2, n_samples=50):
    """Plot ECDFs of permutation samples and original datasets."""
    fig, ax = plt.subplots(1, 1)
    fig.set_layout_engine('tight')

    for i in range(n_samples):
        perm_sample_1, perm_sample_2 = permutation_sample(data1, data2)
        x_1, y_1 = eda_utils.ecdf(perm_sample_1)
        x_2, y_2 = eda_utils.ecdf(perm_sample_2)
        ax.plot(x_1, y_1, marker='.', linestyle='none', color='red', alpha=0.02)
        ax.plot(x_2, y_2, marker='.', linestyle='none', color='blue', alpha=0.02)

    x_1, y_1 = eda_utils.ecdf(data1)
    x_2, y_2 = eda_utils.ecdf(data2)
    ax.plot(x_1, y_1, marker='.', linestyle='none', color='red')
    ax.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

    plt.show()