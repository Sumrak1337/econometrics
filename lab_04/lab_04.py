import numpy as np
import statsmodels.api as sm

xs = np.array([184.71, 243.42, 294.55, 323.04, 341.88, 410.25,
               403.18, 422.73, 382.31, 430.27, 524.51, 521.04])

ys = np.array([158.33, 191.28, 228.73, 280.92, 270.18, 346.72,
               390.87, 402.15, 346.41, 385.14, 464.75, 456.83])

n = len(xs)  # n = m

xs_diffs = np.array([xs[i] - xs[i - 1] for i in range(1, n)])
ys_diffs = np.array([ys[i] - ys[i - 1] for i in range(1, n)])

m = len(xs_diffs)
# TODO: also add demography stats

alpha = 0.05

# kolm_smirn
# TODO: probably, i need to check, that empirical function is building correct

F_n = sm.distributions.ECDF(xs_diffs)
G_m = sm.distributions.ECDF(ys_diffs)

D_mn1 = np.max(np.array([r / m - F_n(ys_diffs[r - 1]) for r in range(1, m + 1)]))
D_mn1_ = np.max(np.array([G_m(xs_diffs[s - 1]) - (s - 1) / m for s in range(1, m + 1)]))
D_mn2 = np.max(np.array([F_n(ys_diffs[r - 1]) - (r - 1) / m for r in range(1, m + 1)]))
D_mn2_ = np.max(np.array([s / m - G_m(xs_diffs[s - 1]) for s in range(1, m + 1)]))

D_mn = max(D_mn1, D_mn2)
D_mn_ = max(D_mn1_, D_mn2_)

ch_D_mn = np.sqrt(m ** 2 / (2 * m)) * D_mn
ch_D_mn_ = np.sqrt(m ** 2 / (2 * m)) * D_mn_

k = 1.36  # Table

if ch_D_mn > k:
    print('H0 принимаем')
else:
    print('H0 отвергаем')

# TODO: wilk
# TODO: check this

