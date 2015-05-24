#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

from gcca import GCCA
import numpy as np
import logging
from sklearn.datasets import load_digits
from sklearn.datasets import load_boston

import matplotlib.pyplot as plt

class BridgedCCA(GCCA):

    def __init__(self, n_components=2, reg_param=0.1):
        GCCA.__init__(self, n_components, reg_param)

    def fit(self, x0_pair0, x1_pair0, x1_pair1, x2_pair1, x0_pair2, x1_pair2, x2_pair2):

        data_num = 3

        # get list of each modality data (for calculation of variances)
        x0_list = [x0_pair0, x0_pair2]
        x1_list = [x1_pair0, x1_pair1, x1_pair2]
        x2_list = [x2_pair1, x2_pair2]
        x0_all = np.vstack(x0_list)
        x1_all = np.vstack(x1_list)
        x2_all = np.vstack(x2_list)
        all_list = [x0_all, x1_all, x2_all]

        # get list of each pair (for calculation of covariances)
        list01 = [np.vstack([x0_pair0, x0_pair2]), np.vstack([x1_pair0, x1_pair2])]
        list12 = [np.vstack([x1_pair1, x1_pair2]), np.vstack([x2_pair1, x2_pair2])]
        list02 = [x0_pair2, x2_pair2]

        self.logger.info("normalizing")
        # calc mean
        mean_list = [np.mean(x, axis=0) for x in all_list]
        # normalize
        norm_list = [ x - m for x, m in zip(all_list, mean_list)]
        norm_list01 = [ x - m for x, m in zip(list01, [mean_list[0], mean_list[1]])]
        norm_list12 = [ x - m for x, m in zip(list12, [mean_list[1], mean_list[2]])]
        norm_list02 = [ x - m for x, m in zip(list02, [mean_list[0], mean_list[2]])]

        self.logger.info("calc variances")
        var_mat_list = [np.cov(norm_data.T) for norm_data in norm_list]
        self.logger.info("adding regularization term")
        for i, v in enumerate(var_mat_list):
            var_mat_list[i] += self.reg_param * np.average(np.diag(v)) * np.eye(v.shape[0])

        self.logger.info("calc covariances")
        cov_mat_list01 = self.calc_cov_mat(norm_list01)
        cov_mat_list12 = self.calc_cov_mat(norm_list12)
        cov_mat_list02 = self.calc_cov_mat(norm_list02)

        c00 = var_mat_list[0]
        c01 = cov_mat_list01[0][1]
        c11 = var_mat_list[1]
        c12 = cov_mat_list12[0][1]
        c22 = var_mat_list[2]
        c02 = cov_mat_list02[0][1]

        cov_mat = [[np.array([]) for col in range(data_num)] for row in range(data_num)]
        cov_mat[0][0], cov_mat[0][1], cov_mat[0][2] = c00, c01, c02
        cov_mat[1][0], cov_mat[1][1], cov_mat[1][2] = c01.T, c11, c12
        cov_mat[2][0], cov_mat[2][1], cov_mat[2][2] = c02.T, c12.T, c22

        # print c00.shape
        # print c01.shape
        # print c02.shape
        # print c01.T.shape
        # print c11.T.shape
        # print c12.shape
        # print c02.T.shape
        # print c12.T.shape
        # print c22.shape
        self.logger.info("calculating generalized eigenvalue problem ( A*u = (lambda)*B*u )")
        # left = A, right = B
        left = 0.5 * np.vstack([
            np.hstack([np.zeros_like(c00), c01, c02]),
            np.hstack([c01.T, np.zeros_like(c11), c12]),
            np.hstack([c02.T, c12.T, np.zeros_like(c22)])
        ])
        right = np.vstack([
            np.hstack([c00, np.zeros_like(c01), np.zeros_like(c02)]),
            np.hstack([np.zeros_like(c01.T), c11, np.zeros_like(c12)]),
            np.hstack([np.zeros_like(c02.T), np.zeros_like(c12.T), c22])
        ])

        # calc GEV
        self.logger.info("solving")
        eigvals, eigvecs = self.solve_eigprob(left, right)
        d_list = [0] + [sum([len(x.T) for x in all_list][:i + 1]) for i in xrange(data_num)]
        h_list = [eigvecs[start:end] for start, end in zip(d_list[0:-1], d_list[1:])]
        h_list_norm = [ self.eigvec_normalization(h, cov_mat[i][i]) for i, h in enumerate(h_list)]

        # substitute local variables for member variables
        self.data_num = data_num
        self.cov_mat = cov_mat
        self.h_list = h_list_norm
        self.eigvals = eigvals
        self.mean_list = mean_list


def main():

    # set log level
    logging.root.setLevel(level=logging.INFO)

    # create data in advance
    digit = load_digits()
    print digit.data.shape
    a = digit.data[:1000, 0::3]
    b = digit.data[:1000, 1::3]
    c = digit.data[:1000, 2::3]
    # a = np.random.rand(1000, 70)
    # b = np.random.rand(1000, 60)
    # c = np.random.rand(1000, 50)

    # create instance of BridgedCCA
    bcca = BridgedCCA(reg_param=0.001)

    cor_results = []
    # calculate BridgedCCA
    for i in xrange(0, 1000, 50):
        sep1 = i
        sep0 = sep1/2
        sep2 = 1000
        bcca.fit(a[:sep0], b[:sep0], b[sep0:sep1], c[sep0:sep1], a[sep1:sep2], b[sep1:sep2], c[sep1:sep2])
        # transform
        sep3 = sep2
        bcca.transform(a[:sep3], b[:sep3], c[:sep3])
        # # save
        # bcca.save_params("save/bcca.h5")
        # # load
        # bcca.load_params("save/bcca.h5")
        # calc correlations
        pair_list, cor_list = bcca.get_correlations()
        cor_results.append(cor_list)
        # # plot
        # bcca.plot_result()

    try__num = len(cor_results)
    cor_mat = np.array(cor_results)
    plt.plot(range(0, 1000, 50), cor_mat[:, 0],"r-")
    plt.plot(range(0, 1000, 50), cor_mat[:, 1], "g-")
    plt.plot(range(0, 1000, 50), cor_mat[:, 2], "b-")
    plt.show()

if __name__=="__main__":

    main()
