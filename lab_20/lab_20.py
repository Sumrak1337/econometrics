import numpy as np
from data.lab_data import LabData
# TODO: add nice report


class Lab20:
    def __init__(self, main_data: dict):
        self.data = np.array([values for values in main_data.values()])
        self.x1 = self.data[:9, :]
        self.x2 = self.data[9:, :]

    def processing(self):
        x1_means = np.mean(self.x1, axis=0)
        x2_means = np.mean(self.x2, axis=0)

        x1_diffs = self.x1 - x1_means
        x2_diffs = self.x2 - x2_means

        s = (x1_diffs.T @ x1_diffs + x2_diffs.T @ x2_diffs) / (len(self.x1) + len(self.x2) - 2)
        s_inv = np.linalg.inv(s)

        beta = s_inv @ (x1_means - x2_means)
        d2 = (x1_means - x2_means).T @ s_inv @ (x1_means - x2_means)

        y1 = beta @ self.x1.T
        y2 = beta @ self.x2.T
        yt = 0.5 * beta.T @ (x1_means + x2_means)

        class1 = np.array(y1 >= yt).astype(int)
        class2 = np.array(y2 >= yt).astype(int)


dict_data = LabData.lab_20()
Lab20(dict_data).processing()
