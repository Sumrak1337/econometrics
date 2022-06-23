import numpy as np
from data.lab_data import LabData
# TODO: add nice report


class Lab21:
    def __init__(self, main_data):
        self.data = np.array([values for values in main_data.values()])

    def processing(self):
        covariance = np.cov(self.data.T)
        correlation = np.corrcoef(self.data.T)

        self.pca(covariance)
        self.pca(correlation)

    def pca(self, matrix):
        w, v = np.linalg.eig(matrix)
        y = self.data @ v
        explained_disp = w / np.sum(w)


dict_data = LabData.lab_21()
Lab21(dict_data).processing()
