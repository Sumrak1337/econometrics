import numpy as np


class LabData:

    def __init__(self):
        self.lab_03_data = self.lab_03()
        self.lab_04_data = self.lab_04()
        self.lab_05_data = self.lab_05()
        self.lab_06_data = self.lab_06()
        self.lab_08_data = self.lab_08()
        self.lab_10_data = self.lab_10()
        self.lab_13_data = self.lab_13()
        self.lab_19_data = self.lab_19()
        self.lab_20_data = self.lab_20()
        self.lab_21_data = self.lab_21()

    @staticmethod
    def lab_03():
        data = {}
        keys = np.array([year for year in range(1970, 1998)])
        values = np.array([4.14, 4.21, 4.91, 6.47, 5.33, 4.41, 5.29,
                           5.13, 7.12, 9.26, 8.71, 8.69, 9.91, 9.61,
                           9.71, 8.91, 7.73, 8.13, 8.26, 8.72, 9.41,
                           9.09, 9.01, 9.37, 8.78, 8.46, 8.53, 8.67])
        for key, value in zip(keys, values):
            data[key] = value

        return data

    @staticmethod
    def lab_04():
        data = {}
        keys = np.array([year for year in range(1985, 1997)])
        values = np.array([[184.71, 158.33],
                           [243.42, 191.28],
                           [294.55, 228.73],
                           [323.04, 280.92],
                           [341.88, 270.18],
                           [410.25, 346.72],
                           [403.18, 390.87],
                           [422.73, 402.15],
                           [382.31, 346.41],
                           [430.27, 385.14],
                           [524.51, 464.75],
                           [521.04, 456.83]])
        for key, value in zip(keys, values):
            data[key] = value

        return data

    @staticmethod
    def lab_05():
        data = {}
        keys = np.array([year for year in range(1961, 1996)])
        values = np.array([[202.31, 209.81],
                           [219.08, 221.57],
                           [239.85, 248.39],
                           [278.27, 283.73],
                           [306.39, 305.59],
                           [328.61, 337.14],
                           [352.37, 351.59],
                           [402.38, 400.18],
                           [483.37, 474.39],
                           [562.65, 533.68],
                           [609.19, 581.12],
                           [683.46, 633.35],
                           [846.47, 811.16],
                           [1116.27, 1109.35],
                           [1065.21, 1061.39],
                           [1266.58, 1261.47],
                           [1474.85, 1499.88],
                           [1540.11, 1570.85],
                           [1798.81, 1866.38],
                           [2026.65, 2125.11],
                           [2286.64, 2357.69],
                           [2640.57, 2694.44],
                           [2924.28, 2864.72],
                           [3337.84, 3277.94],
                           [3479.21, 3379.87],
                           [3367.92, 3187.85],
                           [3477.43, 3334.15],
                           [3900.28, 3719.21],
                           [4498.77, 4320.59],
                           [4660.38, 4506.95],
                           [4846.69, 4658.41],
                           [4980.87, 4713.19],
                           [5012.75, 4674.73],
                           [5491.28, 5108.09],
                           [5764.45, 5377.49]])
        for key, value in zip(keys, values):
            data[key] = value

        return data

    @staticmethod
    def lab_06():
        data = {}
        keys = np.array([year for year in range(1960, 1991)])
        values = np.array([65.6, 68.1, 70.4, 73.3, 76.5,
                           78.6, 81.0, 83.0, 85.4, 85.9,
                           87.0, 90.2, 92.6, 95.0, 93.3,
                           95.5, 98.3, 99.8, 100.4, 99.3,
                           98.6, 99.9, 100., 102.2, 104.6,
                           106.1, 108.3, 109.4, 110.4, 109.5, 109.7])
        for key, value in zip(keys, values):
            data[key] = value

        return data

    @staticmethod
    def lab_08():
        data = {}
        keys = np.array([1, 2, 3])
        values = np.array([[10.1, 7.3, 5.6, 6.2, 8.4, 8.1, 8., 7.6, 5.3, 7.2],
                           [11.7, 12.2, 11.8, 7.8, 8.9, 9.9, 12.4, 11., 10.3, 13.8, 10.5, 9.8, 9.1],
                           [10.2, 12., 8.8, 8.7, 10.5, 11., 9.1]])
        for key, value in zip(keys, values):
            data[key] = value

        return data

    @staticmethod
    def lab_10():
        data = {}
        keys = np.array(['s_lit', 'g_lit', 'm_lit'])
        values = np.array([[7.45, 8.23, 8.61, 7.12],
                           [6.73, 6.85, 7.55, 6.58],
                           [5.41, 6.13, 5.57, 3.73]])
        for key, value in zip(keys, values):
            data[key] = value

        return data

    @staticmethod
    def lab_13():
        data = {}
        keys = np.array(['dlt', 'N', 'remission'])
        values = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
                           [0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        for key, value in zip(keys, values):
            data[key] = value

        return data

    @staticmethod
    def lab_19():
        data = {}
        keys = np.array(['not bad', 'good', 'great'])
        values = np.array([[[106000, 843],
                            [107500, 907],
                            [75500, 824],
                            [88000, 672],
                            [83000, 698],
                            [52980, 723]],

                           [[146500, 1317],
                            [128500, 1080],
                            [179000, 1688],
                            [219770, 1738],
                            [139000, 1040],
                            [131000, 1404],
                            [197500, 1558],
                            [116000, 976]],

                           [166500, 1318],
                           [275000, 1507],
                           [164500, 1292],
                           [333000, 2350],
                           [330000, 1732],
                           [302120, 2010],
                           [229000, 1749],
                           [201800, 1762]])
        for key, value in zip(keys, values):
            data[key] = value

        return data

    @staticmethod
    def lab_20():
        data = {}
        keys = np.array(['rice', 'tea', 'sugar', 'flour', 'coffee', 'potato', 'butter',
                         'cheese', 'beef', 'petrol', 'lead', 'cast iron', 'copper',
                         'zinc', 'tin', 'rubber', 'mercury', 'copper sheet', 'iron'])
        values = np.array([[72, 50, 8, 0.5],
                           [66.5, 48, 15, 1],
                           [54, 57, 14, 1],
                           [67, 60, 15, 0.9],
                           [44, 57, 14, 0.3],
                           [41, 52, 18, 1.9],
                           [34.5, 50, 4, 0.5],
                           [34.5, 46, 8.5, 1],
                           [24, 54, 3, 1.2],
                           [57, 57, 12.5, 0.9],
                           [100, 54, 17, 0.5],
                           [100, 32, 16.5, 0.7],
                           [96.5, 65, 20.5, 0.9],
                           [79, 51, 18, 0.9],
                           [78, 53, 18, 1.2],
                           [48, 40, 21, 1.6],
                           [155, 44, 20.5, 1.4],
                           [84, 64, 13, 0.8],
                           [105, 35, 17, 1.8]])
        for key, value in zip(keys, values):
            data[key] = value

        return data

    @staticmethod
    def lab_21():
        data = {}
        keys = np.array([year for year in range(2005, 2015)])
        values = np.array([[84, 17.2, 389.4, 23],
                           [82.6, 19.1, 376.3, 20],
                           [83.3, 19.1, 361.3, 27.3],
                           [85.1, 18.7, 357.1, 31.1],
                           [82.6, 17.8, 358.3, 35.7],
                           [77.4, 17.6, 351.5, 40.3],
                           [73, 15.4, 335.8, 47.1],
                           [68.2, 13.9, 328.3, 52.9],
                           [63.1, 12.6, 320.2, 57],
                           [59.4, 14.5, 308.3, 63.3]])
        for key, value in zip(keys, values):
            data[key] = value

        return data

