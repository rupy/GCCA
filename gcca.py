#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import numpy as np
from scipy.linalg import eig
import logging
import os
import sys


class GCCA:

    def __init__(self, n_components=2, reg_param=0.1, calc_time=False):

        # log setting
        program = os.path.basename(sys.argv[0])
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

        self.reg_param = reg_param
        self.x_1 = None
        self.x_2 = None
        self.x_3 = None
        self.weights_1 = None
        self.eigvals_1 = None
        self.weights_2 = None
        self.eigvals_2 = None
        self.weights_3 = None
        self.eigvals_3 = None

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

        z = np.vstack((x_1, x_2, x_3))
        cov = np.cov(z)
        d1 = len(x_1.T)
        d2 = len(x_2.T) + d1
        c11 = cov[:d1, :d1]
        c22 = cov[d1:d2, d1:d2]
        c33 = cov[d2:, d2:]
        c12 = cov[:d1, d1:d2]
        c23 = cov[d1:d2, d2:]
        c13 = cov[:d1, d2:]

        self.logger.info("calculating generalized eigenvalue problem ( A*u = (lambda)*B*u )")
        self.logger.info("adding regularization term")
        c11 += self.reg_param * np.average(np.diag(c11)) * np.eye(c11.shape[0])
        c22 += self.reg_param * np.average(np.diag(c22)) * np.eye(c12.shape[0])
        c33 += self.reg_param * np.average(np.diag(c33)) * np.eye(c33.shape[0])
        c12 += self.reg_param * np.average(np.diag(c12)) * np.eye(c12.shape[0])
        c23 += self.reg_param * np.average(np.diag(c23)) * np.eye(c23.shape[0])
        c13 += self.reg_param * np.average(np.diag(c13)) * np.eye(c13.shape[0])

        # left = A, right = B
        # 1
        self.logger.info("solving")
        left_1 = 0.5 * (c12 + c13)
        right_1 = c11
        eigvals_1, eigvecs_1 = self.solve_eigprob(left_1, right_1)

        # 2
        left_2 = 0.5 * (c12.T + c23)
        right_2 = c22
        eigvals_2, eigvecs_2 = self.solve_eigprob(left_2, right_2)

        # 3
        left_3 = 0.5 * (c13.T + c23.T)
        right_3 = c33
        eigvals_3, eigvecs_3 = self.solve_eigprob(left_3, right_3)

        self.weights_1 = eigvecs_1
        self.eigvals_1 = eigvals_1
        self.weights_2 = eigvecs_2
        self.eigvals_2 = eigvals_2
        self.weights_3 = eigvecs_3
        self.eigvals_3 = eigvals_3

    def transform(self, x_1, x_2, x_3):

        self.logger.info("Normalizing")
        x_1 = self.normalize(x_1)
        x_2 = self.normalize(x_2)
        x_3 = self.normalize(x_3)

        self.logger.info("transform matrices by CCA")
        z_1 = np.dot(x_1, self.weights_1)
        z_2 = np.dot(x_2, self.weights_2)
        z_3 = np.dot(x_3, self.weights_3)

        self.z_1 = z_1
        self.z_2 = z_2
        self.z_3 = z_3

        return z_1, z_2, z_3

    def normalize(self, mat):
        m = np.mean(mat, axis=0)
        mat = mat - m
        return mat


