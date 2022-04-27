import numpy as np
import matplotlib.pyplot as plt


def gauss_distribution(x, mu=0, sigma=1):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def statistical_significance(r, n):
    return r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2)


# 1 # TODO: add random state
np.random.seed(42)
sample1 = np.random.uniform(0, 1, 140)
sample2 = gauss_distribution(sample1, 10, 2)

# 2
sample3 = np.array([np.random.normal(10, 2) for _ in range(140)])

# 3 # TODO: найти критическую точку из распределения стьюдента
r1 = np.corrcoef(sample1, sample2)[0, 1]
r2 = np.corrcoef(sample2, sample3)[0, 1]
r3 = np.corrcoef(sample3, sample1)[0, 1]

stat_signif1 = statistical_significance(r1, len(sample1))
stat_signif2 = statistical_significance(r2, len(sample2))
stat_signif3 = statistical_significance(r3, len(sample3))

# 4
cov1 = np.cov(sample1, sample2)[0, 1]
cov2 = np.cov(sample2, sample3)[0, 1]
cov3 = np.cov(sample3, sample1)[0, 1]

std1 = np.std(sample1, ddof=1)
std2 = np.std(sample2, ddof=1)
std3 = np.std(sample3, ddof=1)

# TODO: check with corrcoef

# 5
corr_matrix = np.corrcoef(np.array([sample1, sample2, sample3]))

# TODO: write a report
