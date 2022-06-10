import numpy as np
import statsmodels.api as sm
import scipy.stats as ss

from data.lab_data import LabData


def kolmogorov(edf):
    left_value = np.max(np.array([i / n - edf.y[i] for i in range(1, n + 1)]))
    right_value = np.max(np.array([edf.y[i] - (i - 1) / n for i in range(1, n + 1)]))
    dn = max(left_value, right_value)
    dn = dn * (np.sqrt(n) - 0.01 + 0.85 / np.sqrt(n))

    # report
    print('Критерий Колмогорова')

    for alpha in alphas:
        answer = False
        d_star = mod_quantiles[alpha][0]

        if dn <= d_star:
            answer = True

        if answer:
            print(f'Гипотеза H0 принимается на уровне значимости {alpha}, так как значение {dn:.8} не попадает в интервал ({d_star}; +inf)')
        else:
            print(f'Гипотеза H0 отвергается на уровне значимости {alpha}, так как значение {dn:.8} попадает в интервал ({d_star}; +inf)')
    print()


def omega(edf):
    s = 0
    for i in range(1, n + 1):
        s += (edf.y[i] - (2 * i - 1) / (2 * n)) ** 2
    omega_sqr = 1 / (12 * n) + s

    omega_sqr = omega_sqr * (1 + 0.5 / n)

    # report
    print('Критерий omega^2')

    for alpha in alphas:
        answer = False
        omega_star = mod_quantiles[alpha][1]

        if omega_sqr <= omega_star:
            answer = True

        if answer:
            print(f'Гипотеза H0 принимается на уровне значимости {alpha}, так как значение {omega_sqr:.8} не попадает в интервал ({omega_star}; +inf)')
        else:
            print(f'Гипотеза H0 отвергается на уровне значимости {alpha}, так как значение {omega_sqr:.8} попадает в интервал ({omega_star}; +inf)')


dict_data = LabData().lab_03()

data = np.array([])
for value in dict_data.values():
    data = np.append(data, value)

alphas = [0.05, 0.1]
mod_quantiles = {0.15: [0.775, 0.091],
                 0.10: [0.819, 0.104],
                 0.05: [0.895, 0.126],
                 0.025: [0.955, 0.148],
                 0.01: [1.035, 0.178]}

diffs = np.array([data[i] - data[i - 1] for i in range(1, len(data))])
diffs = np.unique(diffs.round(decimals=2))
n = len(diffs)

mean = np.mean(diffs)
var = np.var(diffs, ddof=1)

min_diffs = np.min(diffs)
max_diffs = np.max(diffs)
cum_freq = ss.cumfreq(diffs, numbins=n-1, defaultreallimits=(min_diffs, max_diffs))
cum_freq = np.append([1], cum_freq[0])
freq = cum_freq / n

intervals = np.linspace(min_diffs, max_diffs, n)
ecdf = sm.distributions.StepFunction(intervals, freq, side='right')

kolmogorov(ecdf)
omega(ecdf)
