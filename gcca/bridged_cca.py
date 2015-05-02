#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

from gcca import GCCA
import numpy as np
import logging
from sklearn.datasets import load_digits

class BridgedCCA(GCCA):

    def __init__(self, n_components=2, reg_param=0.1):
        GCCA.__init__(self, n_components, reg_param)

    def fit(self, x0_pair0, x1_pair0, x1_pair1, x2_pair1):

        p0_list = [x0_pair0, x1_pair0]
        p1_list = [x1_pair1, x2_pair1]
        data_num = 3

        # data size check
        p0_num = len(p0_list)
        p1_num = len(p1_list)
        self.logger.info("pair0 data num is %d", p0_num)
        for i, x in enumerate(p0_list):
            self.logger.info("pair0 data shape x_%d: %s", i, x.shape)
        self.logger.info("pair1 data num is %d", p1_num)
        for i, x in enumerate(p1_list):
            self.logger.info("pair1 data shape x_%d: %s", i + 1, x.shape)

        self.logger.info("normalizing")
        p0_norm_list = [ self.normalize(x) for x in p0_list]
        p1_norm_list = [ self.normalize(x) for x in p1_list]

        p0_d_list = [0] + [sum([len(x.T) for x in p0_list][:i + 1]) for i in xrange(p0_num)]
        p1_d_list = [0] + [sum([len(x.T) for x in p1_list][:i + 1]) for i in xrange(p1_num)]

        p0_cov_mat = self.calc_cov_mat(p0_norm_list)
        p0_cov_mat = self.add_regularization_term(p0_cov_mat)
        p1_cov_mat = self.calc_cov_mat(p1_norm_list)
        p1_cov_mat = self.add_regularization_term(p1_cov_mat)
        self.logger.info("calc variance")
        x1_all = np.vstack([x1_pair0, x1_pair1])
        x1_var = np.cov(x1_all.T)
        self.logger.info("adding regularization term")
        x1_var += self.reg_param * np.average(np.diag(x1_var)) * np.eye(x1_var.shape[0])

        x_list = [x0_pair0, x1_all, x2_pair1]
        d_list = [0] + [sum([len(x.T) for x in x_list][:i + 1]) for i in xrange(data_num)]

        c00 = p0_cov_mat[0][0]
        c01 = p0_cov_mat[0][1]
        # c11 = p0_cov_mat[1][1]
        # c11 = p1_cov_mat[1 - 1][1 - 1]
        c11 = x1_var
        c12 = p1_cov_mat[1 - 1][2 - 1]
        c22 = p1_cov_mat[2 - 1][2 - 1]
        c02 = np.zeros((c00.shape[0], c22.shape[1]))

        cov_mat = [[np.array([]) for col in range(data_num)] for row in range(data_num)]
        cov_mat[0][0], cov_mat[0][1], cov_mat[0][2] = c00, c01, c02
        cov_mat[1][0], cov_mat[1][1], cov_mat[1][2] = c01.T, c11, c12
        cov_mat[2][0], cov_mat[2][1], cov_mat[2][2] = c02.T, c12.T, c22

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
        h_list = [eigvecs[start:end] for start, end in zip(d_list[0:-1], d_list[1:])]
        h_list_norm = [ self.eigvec_normalization(h, cov_mat[i][i]) for i, h in enumerate(h_list)]

        # substitute local variables for member variables
        self.data_num = data_num
        self.cov_mat = cov_mat
        self.h_list = h_list_norm
        self.eigvals = eigvals

def main():

    # set log level
    logging.root.setLevel(level=logging.INFO)

    # create data in advance
    digit = load_digits()
    a = digit.data[:150, 0::3]
    b = digit.data[:150, 1::3]
    c = digit.data[:150, 2::3]
    # a = np.random.rand(100, 50)
    # b = np.random.rand(100, 60)
    # c = np.random.rand(100, 70)

    # create instance of BridgedCCA
    bcca = BridgedCCA(reg_param=0.0001)
    # calculate BridgedCCA
    bcca.fit(a[:50], b[:50], b[50:100], c[50:100])
    # transform
    bcca.transform(a[100:], b[100:], c[100:])
    # save
    bcca.save_params("save/bcca.h5")
    # load
    bcca.load_params("save/bcca.h5")
    # calc correlations
    bcca.calc_correlations()
    # plot
    bcca.plot_result()

if __name__=="__main__":

    main()
