#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

from gcca import GCCA
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from matplotlib import colors
import h5py

class CCA(GCCA):

    def __init__(self, n_components=2, reg_param=0.1):
        GCCA.__init__(self, n_components, reg_param)

        # log setting
        program = os.path.basename(__name__)
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s')

        self.z_p = np.array([])

    def fit(self, x0, x1):

        x_list = [x0, x1]

        # data size check
        data_num = len(x_list)
        self.logger.info("data num is %d", data_num)
        for i, x in enumerate(x_list):
            self.logger.info("data shape x_%d: %s", i, x.shape)

        self.logger.info("normalizing")
        x_norm_list = [ self.normalize(x) for x in x_list]

        d_list = [0] + [sum([len(x.T) for x in x_list][:i + 1]) for i in xrange(data_num)]
        cov_mat = self.calc_cov_mat(x_norm_list)
        cov_mat = self.add_regularization_term(cov_mat)
        c_00 = cov_mat[0][0]
        c_01 = cov_mat[0][1]
        c_11 = cov_mat[1][1]

        self.logger.info("calculating generalized eigenvalue problem ( A*u = (lambda)*B*u )")

        # 1
        left_1 = np.dot(c_01, np.linalg.solve(c_11,c_01.T))
        right_1 = c_00
        eigvals_1, eigvecs_1 = self.solve_eigprob(left_1, right_1)
        eigvecs_1_norm = self.eigvec_normalization(eigvecs_1, right_1)
        # 2
        right_2 = c_11
        eigvecs_2 = 1 / eigvals_1 * np.dot(np.linalg.solve(c_11, c_01.T), eigvecs_1_norm)
        eigvecs_2_norm = self.eigvec_normalization(eigvecs_2, right_2)

        # substitute local variables for member variables
        self.data_num = data_num
        self.cov_mat = cov_mat
        self.h_list = [eigvecs_1_norm, eigvecs_2_norm]
        self.eigvals = eigvals_1

    def ptransform(self, x0, x1, beta=0.5):

        x0_projected, x1_projected = self.transform(x0, x1)

        I = np.eye(len(self.eigvals))
        lamb = np.diag(self.eigvals)
        mat1 = np.linalg.solve(I - np.diag(self.eigvals**2), I)
        mat2 = -np.dot(mat1, lamb)
        mat12 = np.vstack((mat1, mat2))
        mat21 = np.vstack((mat2, mat1))
        mat = np.hstack((mat12, mat21))
        p = np.vstack((lamb**beta, lamb**(1-beta)))
        q = np.vstack((x0_projected.T, x1_projected.T))
        z = np.dot(p.T, np.dot(mat, q)).T[:,:self.n_components]

        self.z_p = z

        return x0_projected, x1_projected, z

    def save_params(self, filepath):

        GCCA.save_params(self, filepath)
        if len(self.z_p) != 0:
            with h5py.File(filepath, 'a') as f:
                f.create_dataset("z_p", data=self.z_p)
                f.flush()

    def load_params(self, filepath):

        GCCA.load_params(self, filepath)

        with h5py.File(filepath, "r") as f:
            if "z_p" in f:
                self.z_p = f["z_p"].value
            f.flush()

    def plot_result(self):

        self.logger.info("plotting result")
        row_num = 2
        col_num = 2

        # begin plot
        plt.figure()

        color_list = colors.cnames.keys()
        plt.subplot(row_num, col_num, 1)
        plt.plot(self.z_list[0][:, 0], self.z_list[0][:, 1], c=color_list[0], marker='.', ls=' ')
        plt.title("Z_0(CCA)")
        plt.subplot(row_num, col_num, 2)
        plt.plot(self.z_list[1][:, 0], self.z_list[1][:, 1], c=color_list[1], marker='.', ls=' ')
        plt.title('Z_1(CCA)')

        plt.subplot(row_num, col_num, 3)
        plt.plot(self.z_list[0][:, 0], self.z_list[0][:, 1], c=color_list[0], marker='.', ls=' ')
        plt.plot(self.z_list[1][:, 0], self.z_list[1][:, 1], c=color_list[1], marker='.', ls=' ')
        plt.title('Z_ALL(CCA)')

        if len(self.z_p) != 0:
            plt.subplot(row_num, col_num, 4)
            plt.plot(self.z_p[:, 0], self.z_p[:, 1], c=color_list[2], marker='.', ls=' ')
            plt.title('Z(PCCA)')

        plt.show()
        
def main():

    # set log level
    logging.root.setLevel(level=logging.INFO)

    # create data in advance
    a = np.random.rand(50, 50)
    b = np.random.rand(50, 60)

    # create instance of CCA
    cca = CCA()
    # calculate CCA
    cca.fit(a, b)
    # transform
    cca.transform(a, b)
    # transform by PCCA
    cca.ptransform(a, b)
    # save
    cca.save_params("save/cca.h5")
    # load
    cca.load_params("save/cca.h5")
    # plot
    cca.plot_result()
    # calc correlations
    cca.calc_correlations()

if __name__=="__main__":

    main()