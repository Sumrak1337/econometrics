import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as ss


class Lab02:
    def __init__(self):
        self.rs = np.random.RandomState(42)
        self.a = self.rs.randint(0, 10)
        self.sigma = self.rs.randint(1, 15)
        self.bins1 = 8
        self.bins2 = 12
        self.bins3 = 5
        self.bins4 = 7
        self.colors = list(mcolors.TABLEAU_COLORS)

        self.data = ss.norm.ppf(self.rs.random(200), loc=self.a, scale=self.sigma)

        self.disp = np.var(self.data, ddof=1)
        self.sd = np.std(self.data, ddof=1)
        self.mean = np.mean(self.data)

        self.sample1 = self.rs.choice(self.data, 30)
        self.mean1 = np.mean(self.sample1)
        self.disp1 = np.var(self.sample1, ddof=1)
        self.sd1 = np.std(self.sample1, ddof=1)

        self.sample2 = self.rs.choice(self.data, 30)
        self.mean2 = np.mean(self.sample2)
        self.disp2 = np.var(self.sample2, ddof=1)
        self.sd2 = np.std(self.sample2, ddof=1)

    def processing(self):
        self.rel_freq_hist(self.data, self.bins1, 'data')
        self.rel_freq_hist(self.data, self.bins2, 'data')
        self.probability()
        self.rel_freq_hist(self.sample1, self.bins3, 'sample1')
        self.rel_freq_hist(self.sample1, self.bins4, 'sample1')
        self.rel_freq_hist(self.sample2, self.bins3, 'sample2')
        self.rel_freq_hist(self.sample2, self.bins4, 'sample2')

        self.mean_conf_interval_known_sd(self.data)
        self.mean_conf_interval_known_sd(self.sample1)
        self.mean_conf_interval_known_sd(self.sample2)

        self.mean_conf_interval_unknown_sd(self.data)
        self.mean_conf_interval_unknown_sd(self.sample1)
        self.mean_conf_interval_unknown_sd(self.sample2)

        self.disp_conf_interval_unknown_mean(self.data)
        self.disp_conf_interval_unknown_mean(self.sample1)
        self.disp_conf_interval_unknown_mean(self.sample2)

        self.general_sd_conf_interval(self.data)
        self.general_sd_conf_interval(self.sample1)
        self.general_sd_conf_interval(self.sample2)

        self.bernoulli()

    def rel_freq_hist(self, data: np.array, bins, name):
        n, intervals, _ = plt.hist(data, bins=bins)
        mid_intervals = np.array([(intervals[i] + intervals[i + 1]) / 2 for i in range(bins)])
        plt.close('all')

        plt.figure()
        plt.title(f'Гистограмма относительных частот ({name})')
        rel_freq = n / len(data)
        plt.bar(mid_intervals, height=rel_freq)
        plt.plot(mid_intervals, rel_freq, color=self.colors[1])
        plt.xlabel(f'x_value ({bins} intervals)')
        plt.ylabel('relative frequency')
        plt.show()
        plt.close('all')

    def probability(self):
        prob1 = ss.norm.pdf(11, loc=self.mean, scale=self.sd) - ss.norm.pdf(5, loc=self.mean, scale=self.sd)
        prob2 = ss.norm.pdf(11, loc=self.a, scale=self.sigma) - ss.norm.pdf(5, loc=self.a, scale=self.sigma)
        print(f'Значение вероятности F(11) - F(5) при вычисленных a = {self.mean:.4} и sigma = {self.sd:.4} равно: {prob1:.5}')
        print(f'Значение вероятности F(11) - F(5) при данных a = {self.a} и sigma = {self.sigma} равно: {prob2:.5}')
        print(f'Абсолютная разность: {np.abs(prob1 - prob2):.4}')
        print()

    @staticmethod
    def mean_conf_interval_known_sd(data):
        left_edge = np.mean(data) - ss.norm(loc=0, scale=1).ppf(0.975) * np.std(data, ddof=1) / np.sqrt(len(data))
        right_edge = np.mean(data) + ss.norm(loc=0, scale=1).ppf(0.975) * np.std(data) / np.sqrt(len(data))
        print(f'Доверительный интервал для оценки мат. ожидания при известном СКО (уровень доверия = 0.95): ({left_edge:.3}, {right_edge:.3})')
        print()

    @staticmethod
    def mean_conf_interval_unknown_sd(data):
        n = len(data)
        left_edge = np.mean(data) - ss.t.ppf(q=0.975, df=n - 1) * np.std(data, ddof=1) / np.sqrt(n)
        right_edge = np.mean(data) + ss.t.ppf(q=0.975, df=n - 1) * np.std(data, ddof=1) / np.sqrt(n)
        print(f'Доверительный интервал для оценки мат.ожидания при неизвестном СКО (уровень доверия = 0.95): ({left_edge:.3}, {right_edge:.3})')
        print()

    @staticmethod
    def disp_conf_interval_unknown_mean(data):
        n = len(data)
        t1 = ss.chi2.ppf(q=1-0.975, df=n-1)
        t2 = ss.chi2.ppf(q=1-0.025, df=n-1)
        left_edge = (n - 1) * np.var(data, ddof=1) / t2
        right_edge = (n - 1) * np.var(data, ddof=1) / t1
        print(f'Доверительный интервал для оценки дисперсии при неизвестном значении генерального среднего (уровень доверия = 0.95): [{left_edge:.3}, {right_edge:.3}]')
        print()

    @staticmethod
    def general_sd_conf_interval(data):
        n = len(data)
        t1 = ss.chi2.ppf(q=1-0.975, df=n-1)
        t2 = ss.chi2.ppf(q=1-0.025, df=n-1)
        left_edge = np.sqrt(n - 1) * np.std(data, ddof=1) / np.sqrt(t2)
        right_edge = np.sqrt(n - 1) * np.std(data, ddof=1) / np.sqrt(t1)
        print(f'Доверительный интервал для оценки генерального СКО (уровень доверия = 0.95): [{left_edge:.3}, {right_edge:.3}]')
        print()

    def bernoulli(self):
        sample = ss.bernoulli.rvs(p=0.3, random_state=self.rs, size=500)
        n = len(sample)
        p = np.sum(sample) / n
        z = ss.norm.ppf(q=0.975, loc=0, scale=1)
        print(z)
        left_edge = p - z * np.sqrt(p * (1 - p)) / np.sqrt(n)
        right_edge = p + z * np.sqrt(p * (1 - p)) / np.sqrt(n)
        print(f'Доверительный интервал для оценки параметра p в распределении Бернулли (уровень доверия = 0.95): ({left_edge:.3}, {right_edge:.3})')
        print()


Lab02().processing()
