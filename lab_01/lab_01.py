import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as ss


class Lab01:
    def __init__(self):
        self.rs = np.random.RandomState(42)
        self.colors = list(mcolors.TABLEAU_COLORS)

        self.a1 = self.rs.randint(0, 10)
        self.a2 = self.rs.randint(0, 10)
        self.sigma1 = self.rs.randint(1, 15)
        self.sigma2 = self.rs.randint(1, 15)

        self.sample1 = ss.norm.ppf(self.rs.random(80), loc=self.a1, scale=self.sigma1).astype(int)
        self.sample2 = ss.norm.ppf(self.rs.random(60), loc=self.a2, scale=self.sigma2).astype(int)
        self.general_sample = np.append(self.sample1, self.sample2)
        self.general_sample = np.sort(self.general_sample)
        self.bins = int(np.ceil(1 + 3.2 * np.log10(len(self.general_sample))))  # 10
        print(self.bins)

        self.n = None
        self.mid_intervals = None
        self.rel_freq = None
        self.rel_dens = None
        self.cum_freq = None
        self.rel_cum_freq = None

    def processing(self):
        self.describe(self.general_sample)
        self.n, self.mid_intervals = self.freq_hist()

        self.rel_freq_hist(self.n)
        self.rel_dens = self.rel_freq / np.diff(self.mid_intervals)[0]
        self.rel_den_hist()

        self.cum_freq = np.cumsum(self.n)
        self.rel_cum_freq = self.cum_freq / len(self.general_sample)
        self.rel_cum_hist()

        self.freq_hist_cum_freq()

    def freq_hist(self):
        plt.figure()
        plt.title('Гистограмма частот распределения')
        n, intervals, _ = plt.hist(self.general_sample, rwidth=0.9, bins=self.bins)
        mid_intervals = np.array([(intervals[i] + intervals[i + 1]) / 2 for i in range(self.bins)])
        plt.plot(mid_intervals, n)
        plt.xlabel(f'x_value ({self.bins} intervals)')
        plt.ylabel('frequency')
        plt.show()
        plt.close('all')

        return n, mid_intervals

    def rel_freq_hist(self, n):
        plt.figure()
        plt.title('Гистограмма относительных частот')
        self.rel_freq = n / len(self.general_sample)
        plt.bar(self.mid_intervals, height=self.rel_freq, width=50/self.bins)
        plt.plot(self.mid_intervals, self.rel_freq, color=self.colors[1])
        plt.xlabel(f'x_value ({self.bins} intervals)')
        plt.ylabel('relative frequency')
        plt.show()
        plt.close('all')

    def rel_den_hist(self):
        plt.figure()
        plt.title('Гистограмма плотности относительных частот')
        plt.bar(self.mid_intervals, height=self.rel_dens, width=50/self.bins)
        plt.plot(self.mid_intervals, self.rel_dens, color=self.colors[1])
        plt.xlabel(f'x_value ({self.bins} intervals)')
        plt.ylabel('frequency density')
        plt.show()
        plt.close('all')

    def rel_cum_hist(self):
        plt.figure()
        plt.title('Гистограмма относительных кумулятивных частот')
        plt.bar(self.mid_intervals, height=self.rel_cum_freq, width=50 / self.bins)
        plt.plot(self.mid_intervals, self.rel_cum_freq, color=self.colors[1])
        plt.xlabel(f'x_value ({self.bins} intervals)')
        plt.ylabel('relative frequency')
        plt.show()
        plt.close('all')

    def freq_hist_cum_freq(self, bins=7):
        n, intervals, _ = plt.hist(self.general_sample, rwidth=0.9, bins=bins)
        plt.close('all')
        plt.figure()
        plt.title('Гистограмма распределения частот и график интегральных частот')
        mid_ints = np.array([(intervals[i] + intervals[i + 1]) / 2 for i in range(bins)])
        cum_freq = np.cumsum(n) / len(self.general_sample)
        plt.bar(mid_ints, height=n / len(self.general_sample), width=50 / bins)
        plt.plot(mid_ints, cum_freq, color=self.colors[1])
        plt.xlabel(f'x_value ({bins} intervals)')
        plt.ylabel('frequency')
        plt.show()
        plt.close('all')

    @staticmethod
    def describe(sample: np.array):
        print('max: ', np.max(sample))
        print('min: ', np.min(sample))
        print('n: ', len(sample))
        print('mean: ', np.mean(sample))
        print('median: ', np.median(sample))
        print('mode: ', ss.mode(sample)[0][0])
        print('range:', np.max(sample) - np.min(sample))
        print('unbiased sample dispersion: ', np.var(sample, ddof=1))
        print('biased sample dispersion: ', np.var(sample))
        print('standard deviation (unbiased): ', np.std(sample, ddof=1))
        print('standard deviation (biased): ', np.std(sample))
        print('mean absolute deviation: ', np.mean(np.abs(sample - np.mean(sample))))
        print('kurtosis: ', ss.kurtosis(sample))
        print('skewness: ', ss.skew(sample))
        print('variation coefficient', np.std(sample, ddof=1) / np.mean(sample))
        print('sampling error: ', np.std(sample, ddof=1) / np.sqrt(len(sample)))


Lab01().processing()
