import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats


def describe(arr: np.array):
    print('max: ', np.max(arr))
    print('min: ', np.min(arr))
    print('n: ', len(arr))
    print('mean: ', np.mean(arr))
    print('median: ', np.median(arr))
    print('mode: ', stats.mode(arr)[0][0])  # TODO: ask about several values
    print('range:', np.max(arr) - np.min(arr))
    print('unbiased sample dispersion: ', np.var(arr, ddof=1))
    print('biased sample dispersion: ', np.var(arr))
    print('standard deviation (unbiased): ', np.std(arr, ddof=1))
    print('standard deviation (biased): ', np.std(arr))
    print('kurtosis: ', stats.kurtosis(arr))
    print('skewness: ', stats.skew(arr))


colors = list(mcolors.TABLEAU_COLORS)

# 1
rs = np.random.RandomState(42)
a = np.array([np.floor(rs.normal(7, 2) + 0.5) for _ in range(80)])
# a = np.array([np.floor(random.gauss(7, 2) + 0.5) for _ in range(80)])
# b = np.array([np.floor(random.gauss(4, 3) + 0.5) for _ in range(60)])
b = np.array([np.floor(rs.normal(4, 3) + 0.5) for _ in range(60)])  # TODO: check

# 2
sample = np.concatenate((a, b))
sample.sort()
variation_series = sample.copy()

# 3 and 4
# describe(variation_series)

unique, counts = np.unique(variation_series, return_counts=True)
# 5  # TODO: add labels
plt.figure()
n, intervals, _ = plt.hist(variation_series, rwidth=0.9)
mid_intervals = np.array([(intervals[i] + intervals[i + 1]) / 2 for i in range(len(intervals) - 1)])
plt.plot(mid_intervals, n)
plt.grid(True)
plt.show()
plt.close('all')

# 6  # TODO: add labels
rel_freqs = n / len(variation_series)
# print(rel_freqs)
plt.bar(mid_intervals, height=rel_freqs)
plt.plot(mid_intervals, rel_freqs, color=colors[1])
plt.grid()
plt.show()
plt.close('all')

# plt.figure()
# plt.show()
