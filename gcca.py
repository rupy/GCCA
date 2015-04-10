#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import numpy as np
from scipy.linalg import eig
import logging
import os
import sys
import matplotlib.pyplot as plt

class GCCA:

    def __init__(self, n_components=2, reg_param=0.1):

        # log setting
        program = os.path.basename(sys.argv[0])
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

        # GCCA params
        self.n_components = n_components
        self.reg_param = reg_param

        # data
        self.x_1 = None
        self.x_2 = None
        self.x_3 = None

        # Result of fitting
        self.c11 = None
        self.c22 = None
        self.c33 = None
        self.c12 = None
        self.c23 = None
        self.c13 = None
        self.h_1 = None
        self.h_2 = None
        self.h_3 = None
        self.eigvals = None

    def solve_eigprob(self, left, right):

        self.logger.info("calculating eigen dimension")
        eig_dim = min([np.linalg.matrix_rank(left), np.linalg.matrix_rank(right)])

        self.logger.info("calculating eigenvalues and eigenvector")
        eig_vals, eig_vecs = eig(left, right)# ;print eig_vals.imag

        self.logger.info("sorting eigenvalues and eigenvector")
        sort_indices = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[sort_indices][:eig_dim].real
        eig_vecs = eig_vecs[:,sort_indices][:,:eig_dim].real

        # regularization
        self.logger.info("regularizing")
        # eig_vecs = np.dot(eig_vecs, np.diag(np.reciprocal(np.linalg.norm(eig_vecs, axis=0))))
        var = np.dot(eig_vecs.T, np.dot(right, eig_vecs))
        # print var
        invvar = np.diag(np.reciprocal(np.sqrt(np.diag(var))))
        # print invvar
        eig_vecs = np.dot(eig_vecs, invvar)

        print np.dot(eig_vecs.T, np.dot(right, eig_vecs)).round().astype(int)

        return eig_vals, eig_vecs


    def fit(self, x_1, x_2, x_3):

        self.x_1 = x_1
        self.x_2 = x_2
        self.x_3 = x_3

        self.logger.info("calculating average, variance, and covariance")
        z = np.vstack((x_1.T, x_2.T, x_3.T))
        print z.shape
        cov = np.cov(z)
        d1 = len(x_1.T)
        d2 = len(x_2.T) + d1
        print d1, d2
        c11 = cov[:d1, :d1]
        c22 = cov[d1:d2, d1:d2]
        c33 = cov[d2:, d2:]
        c12 = cov[:d1, d1:d2]
        c23 = cov[d1:d2, d2:]
        c13 = cov[:d1, d2:]
        # print c11.shape
        # print c22.shape
        # print c33.shape
        # print c12.shape
        # print c23.shape
        # print c13.shape

        self.logger.info("calculating generalized eigenvalue problem ( A*u = (lambda)*B*u )")
        self.logger.info("adding regularization term")
        c11 += self.reg_param * np.average(np.diag(c11)) * np.eye(c11.shape[0])
        c22 += self.reg_param * np.average(np.diag(c22)) * np.eye(c22.shape[0])
        c33 += self.reg_param * np.average(np.diag(c33)) * np.eye(c33.shape[0])

        # left = A, right = B
        self.logger.info("solving")
        left = 0.5 * np.vstack([
            np.hstack([np.zeros_like(c11), c12, c13]),
            np.hstack([c12.T, np.zeros_like(c22), c23]),
            np.hstack([c13.T, c23.T, np.zeros_like(c33)])
        ])
        right = np.vstack([
            np.hstack([c11, np.zeros_like(c12), np.zeros_like(c13)]),
            np.hstack([np.zeros_like(c12.T), c22, np.zeros_like(c23)]),
            np.hstack([np.zeros_like(c13.T), np.zeros_like(c23.T), c33])
        ])
        eigvals, eigvecs = self.solve_eigprob(left, right)

        # substitute local variables for member variables
        self.h_1 = eigvecs[:d1]
        self.h_2 = eigvecs[d1:d2]
        self.h_3 = eigvecs[d2:]
        self.eigvals = eigvals
        self.c11 = c11
        self.c22 = c22
        self.c33 = c33
        self.c12 = c12
        self.c23 = c23
        self.c13 = c13

    def transform(self, x_1, x_2, x_3):

        self.logger.info("Normalizing")
        x_1 = self.normalize(x_1)
        x_2 = self.normalize(x_2)
        x_3 = self.normalize(x_3)

        self.logger.info("transform matrices by GCCA")
        z_1 = np.dot(x_1, self.h_1)
        z_2 = np.dot(x_2, self.h_2)
        z_3 = np.dot(x_3, self.h_3)

        self.z_1 = z_1
        self.z_2 = z_2
        self.z_3 = z_3

        return z_1, z_2, z_3

    def fit_transform(self, x_1, x_2, x_3):
        self.fit(x_1, x_2, x_3)
        self.transform(x_1, x_2, x_3)

    def save_params(self, filepath):
        self.logger.info("saving gcca")
        np.save(filepath + "n_components.npy" , self.n_components)
        np.save(filepath + "reg_param.npy", self.reg_param)
        np.save(filepath + "h_1.npy", self.h_1)
        np.save(filepath + "h_2.npy", self.h_2)
        np.save(filepath + "h_3.npy", self.h_3)
        np.save(filepath + "eigvals.npy", self.eigvals)
        np.save(filepath + "c11.npy", self.c11)
        np.save(filepath + "c22.npy", self.c22)
        np.save(filepath + "c33.npy", self.c33)
        np.save(filepath + "c12.npy", self.c12)
        np.save(filepath + "c23.npy", self.c23)
        np.save(filepath + "c13.npy", self.c13)

    def load_params(self, filepath):
        self.logger.info("loading gcca")
        self.n_components = np.load(filepath + "n_components.npy")
        self.reg_param = np.load(filepath + "reg_param.npy")
        self.h_1 = np.load(filepath + "h_1.npy")
        self.h_2 = np.load(filepath + "h_2.npy")
        self.h_3 = np.load(filepath + "h_3.npy")
        self.eigvals = np.load(filepath + "eigvals.npy")
        self.c11 = np.load(filepath + "c11.npy")
        self.c22 = np.load(filepath + "c22.npy")
        self.c33 = np.load(filepath + "c33.npy")
        self.c12 = np.load(filepath + "c12.npy")
        self.c23 = np.load(filepath + "c23.npy")
        self.c13 = np.load(filepath + "c13.npy")

    def normalize(self, mat):
        m = np.mean(mat, axis=0)
        mat = mat - m
        return mat

    def plot_gcca_result(self):

        # begin plot
        plt.figure()

        plt.subplot(221)
        plt.plot(self.z_1[:, 0], self.z_1[:, 1], '.r')
        plt.title('GCCA Z1(English)')

        plt.subplot(222)
        plt.plot(self.z_2[:, 0], self.z_2[:, 1], '.g')
        plt.title('GCCA Z2(Image)')

        plt.subplot(223)
        plt.plot(self.z_3[:, 0], self.z_1[:, 1], '.b')
        plt.title('GCCA Z3(Japanese)')

        plt.subplot(224)
        plt.plot(self.z_1[:, 0], self.z_1[:, 1], '.r')
        plt.plot(self.z_2[:, 0], self.z_2[:, 1], '.g')
        plt.plot(self.z_3[:, 0], self.z_1[:, 1], '.b')
        plt.title('GCCA Z123(ALL)')

        plt.show()



if __name__=="__main__":
    pass