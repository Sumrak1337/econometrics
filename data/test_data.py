import numpy as np


class TestData01:

    def __init__(self):
        self.test_01_data = self.test_01()
        self.test_02_data = self.test_02()
        self.test_03_data = self.test_03()
        self.test_04_data = self.test_04()
        self.test_05_data = self.test_05()
        self.test_06_data = self.test_06()
        self.test_07_data = self.test_07()
        self.test_08_data = self.test_08()
        self.test_09_data = self.test_09()
        self.test_10_data = self.test_10()

    @staticmethod
    def test_01():
        return ...

    @staticmethod
    def test_02():
        return ...

    @staticmethod
    def test_03():
        return ...

    @staticmethod
    def test_04():
        return ...

    @staticmethod
    def test_05():
        return ...

    @staticmethod
    def test_06():
        return ...

    @staticmethod
    def test_07():
        return ...

    @staticmethod
    def test_08():
        return ...

    @staticmethod
    def test_09():
        return ...

    @staticmethod
    def test_10():
        return ...


class TestData02:
    def __init__(self):
        self.test_11_data = self.test_11()

    @staticmethod
    def test_11():
        data = np.array([[5, 4, 275.7],
                         [8, 6, 285.8],
                         [7, 2, 273.],
                         [10, 5, 419.8],
                         [12, 8, 437.5],
                         [7, 4, 318.9],
                         [11, 7, 567.9],
                         [16, 9, 577.8],
                         [13, 4, 425.8],
                         [5, 2, 318.3],
                         [10, 6, 407.4],
                         [5, 5, 226.],
                         [9, 4, 349.6],
                         [19, 8, 647.2],
                         [20, 6, 528.4],
                         [10, 3, 321.5],
                         [12, 5, 425.8],
                         [7, 3, 347.],
                         [14, 4, 455.4],
                         [14, 5, 425.8],
                         [2, 4, 247.1],
                         [16, 7, 507.3],
                         [9, 5, 378.3],
                         [4, 3, 269.1],
                         [18, 6, 628.6],
                         [18, 5, 457.5],
                         [4, 3, 279.1],
                         [8, 6, 367.],
                         [11, 6, 557.8],
                         [13, 6, 575.]])
        return data


class TestData03:
    ...


class TestData04:
    def __init__(self):
        self.test_04_data = self.test_04()

    @staticmethod
    def test_04():
        data = np.array([37.41682, 59.65335, 56.87352, 46.53041, 45.90191, 59.61057,
                         57.0524, 54.23389, 42.73586, 66.32075, 59.40494, 54.43031,
                         50.67438, 69.10416, 67.38145, 65.65859, 69.97872, 87.96524,
                         90.0366, 91.28878, 86.82967, 103.7924, 111.3326, 107.1626])
        return data
