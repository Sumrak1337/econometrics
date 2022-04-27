import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import norm, t


def gauss_distribution(x, mu_=0, sigma_=1):
    return np.exp(-0.5 * ((x - mu_) / sigma_) ** 2) / (sigma_ * np.sqrt(2 * np.pi))


colors = list(mcolors.TABLEAU_COLORS)

# 1
mu = 7
sigma = 3

np.random.seed(5)
a = np.array([np.floor(np.random.normal(mu, sigma) + 0.5) for _ in range(200)]).astype(int)

plt.figure()

plt.subplot(2, 1, 1)
n8, bins8, _ = plt.hist(a, bins=8, rwidth=0.9, label='8 bins')
mid_bins8 = np.array([(bins8[i] + bins8[i + 1]) / 2 for i in range(len(bins8) - 1)])
plt.plot(mid_bins8, n8, label='relative frequency')
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
n12, bins12, _ = plt.hist(a, bins=12, rwidth=0.9, label='12 bins')
mid_bins12 = np.array([(bins12[i] + bins12[i + 1]) / 2 for i in range(len(bins12) - 1)])
plt.plot(mid_bins12, n12, label='relative frequency')
plt.grid()
plt.legend()
# plt.show()  # TODO: plt.savefigure
plt.close('all')

# 2
disp = np.var(a, ddof=1)
mean = np.mean(a)
sd = np.std(a, ddof=1)

# 3
borders = np.array([5, 11])
prob_general = norm(loc=mu, scale=sigma).cdf(borders[0]) - norm(loc=mu, scale=sigma).cdf(borders[1])
prob_calc = norm(loc=mean, scale=sd).cdf(borders[0]) - norm(loc=mean, scale=sd).cdf(borders[1])

# 4
sample1 = np.random.choice(a, 30)
sample2 = np.random.choice(a, 30)

# 5
sample1_mean = np.mean(sample1)
sample2_mean = np.mean(sample2)
sample1_sd = np.std(sample1, ddof=1)
sample2_sd = np.std(sample2, ddof=1)
sample1_disp = np.var(sample1, ddof=1)
sample2_disp = np.var(sample2, ddof=1)

# TODO: create function
# 6
plt.figure()

plt.subplot(2, 2, 1)
plt.hist(sample1, bins=5, rwidth=0.9, label='sample1, 5 bins')
plt.legend()
plt.grid()

plt.subplot(2, 2, 2)
plt.hist(sample1, bins=7, rwidth=0.9, label='sample1, 7 bins')
plt.legend()
plt.grid()

plt.subplot(2, 2, 3)
plt.hist(sample2, bins=5, rwidth=0.9, label='sample2, 5 bins')
plt.legend()
plt.grid()

plt.subplot(2, 2, 4)
plt.hist(sample2, bins=7, rwidth=0.9, label='sample2, 7 bins')
plt.legend()
plt.grid()

# plt.show() # TODO: plt.savefig

# 7
left_edge1 = mean - norm(loc=0, scale=1).ppf(0.975) * sd / np.sqrt(len(a))
right_edge1 = mean + norm(loc=0, scale=1).ppf(0.975) * sd / np.sqrt(len(a))

left_edge2 = sample1_mean - norm(loc=0, scale=1).ppf(0.975) * sample1_sd / np.sqrt(len(sample1))
right_edge2 = sample1_mean + norm(loc=0, scale=1).ppf(0.975) * sample1_sd / np.sqrt(len(sample1))

left_edge3 = sample2_mean - norm(loc=0, scale=1).ppf(0.975) * sample2_sd / np.sqrt(len(sample2))
right_edge3 = sample2_mean + norm(loc=0, scale=1).ppf(0.975) * sample2_sd / np.sqrt(len(sample2))  # TODO: ask about sigma here

# 8
left_edge11 = mean - t.ppf(q=0.975, df=len(a) - 1) * sd / np.sqrt(len(a))
right_edge11 = mean + t.ppf(q=0.975, df=len(a) - 1) * sd / np.sqrt(len(a))

left_edge22 = sample1_mean - t.ppf(q=0.975, df=len(sample1) - 1) * sample1_sd / np.sqrt(len(sample1))
right_edge22 = sample1_mean + t.ppf(q=0.975, df=len(sample1) - 1) * sample1_sd / np.sqrt(len(sample1))

left_edge33 = sample2_mean - t.ppf(q=0.975, df=len(sample2) - 1) * sample2_sd / np.sqrt(len(sample2))
right_edge33 = sample2_mean + t.ppf(q=0.975, df=len(sample2) - 1) * sample2_sd / np.sqrt(len(sample2))

# 9

