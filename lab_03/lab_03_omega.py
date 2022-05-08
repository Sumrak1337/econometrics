import numpy as np
import statsmodels.api as sm

# TODO: create file with params
# TODO: or create a class with the same functions
data = np.array([4.14, 4.21, 4.91, 6.47, 5.33, 4.41, 5.29,
                 5.13, 7.12, 9.26, 8.71, 8.69, 9.91, 9.61,
                 9.71, 8.91, 7.73, 8.13, 8.26, 8.72, 9.41,
                 9.09, 9.01, 9.37, 8.78, 8.46, 8.53, 8.67])

alphas = np.array([0.05, 0.1])

# first differences
diffs = np.array([data[i] - data[i - 1] for i in range(1, len(data))])
diffs = np.unique(diffs.round(decimals=2))
n = len(diffs)

# 3
mean = np.mean(diffs)
sd = np.std(diffs, ddof=1)

# 4a - empirical function
ecdf = sm.distributions.ECDF(diffs)

# 4b - omega
summ = 0
for i in range(1, n + 1):
    summ += (ecdf(diffs[i - 1]) - (2 * i - 1) / (2 * n)) ** 2
omega = 1 / (12 * n) + summ
# print(omega)
a = np.max(diffs) - np.min(diffs)
# TODO: finish
