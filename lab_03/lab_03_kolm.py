import numpy as np
import statsmodels.api as sm

data = np.array([4.14, 4.21, 4.91, 6.47, 5.33, 4.41, 5.29,
                 5.13, 7.12, 9.26, 8.71, 8.69, 9.91, 9.61,
                 9.71, 8.91, 7.73, 8.13, 8.26, 8.72, 9.41,
                 9.09, 9.01, 9.37, 8.78, 8.46, 8.53, 8.67])
# TODO: add data from statistical processing

alphas = np.array([0.05, 0.1])

# first differences
diffs = np.array([data[i] - data[i - 1] for i in range(1, len(data))])
diffs = np.unique(diffs.round(decimals=2))

# 3
mean = np.mean(diffs)
sd = np.std(diffs, ddof=1)

# 4a - empirical distribution function
# intervals = np.linspace(np.min(diffs), np.max(diffs), 24)
ecdf = sm.distributions.ECDF(diffs)

# 4b - Dn
n = len(diffs)
left_value = np.max(np.array([(i + 1) / n - ecdf(diffs[i]) for i in range(n)]))
right_value = np.max(np.array([ecdf(diffs[i]) - i / n for i in range(n)]))
D_n = max(left_value, right_value)

# 4c - recalculate D_n
D_n = D_n * (np.sqrt(n) - 0.01 + 0.85 / np.sqrt(n))

# 5 - critical interval
answer = False
for alpha in alphas:
    if alpha == 0.05:
        D_star = 0.895
    elif alpha == 0.1:
        D_star = 0.819
    else:
        D_star = 1.035

    if D_n <= D_star:
        answer = True

    if answer:
        print(f'H0 принимается на уровне значимости {alpha}')
    else:
        print(f'H0 отвергается на уровне значимости {alpha}')
