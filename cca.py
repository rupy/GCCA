#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import numpy as np
from scipy.linalg import eig
import time
import logging
import os
import sys
import matplotlib.pyplot as plt

class CCA(object):

    def __init__(self, n_components=2, reg_param=0.1):

        # log setting
        program = os.path.basename(sys.argv[0])
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

        # CCA params
        self.n_components = n_components
        self.reg_param = reg_param

        # data
        self.x = None
        self.y = None

        # Result of fitting
        self.x_weights = None
        self.y_weights = None
        self.eigvals = None
        self.c_xx = None
        self.c_yy = None
        self.c_xy = None

        # transformed data by CCA
        self.x_c = None
        self.y_c = None
        self.z_c = None

        self.y_s = None

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


    def fit(self, x, y):

        self.x = x
        self.y = y

        self.logger.info("calculating average, variance, and covariance")
        z = np.vstack((x.T, y.T))
        cov = np.cov(z)
        p = len(x.T)
        c_xx = cov[:p, :p]
        c_yy = cov[p:, p:]
        c_xy = cov[:p, p:]
        # print c_xx.shape, c_xy.shape, c_yy.shape

        self.logger.info("calculating generalized eigenvalue problem ( A*u = (lambda)*B*u )")
        self.logger.info("adding regularization term")
        c_xx += self.reg_param * np.average(np.diag(c_xx)) * np.eye(c_xx.shape[0])
        c_yy += self.reg_param * np.average(np.diag(c_yy)) * np.eye(c_yy.shape[0])

        self.c_xx = c_xx
        self.c_yy = c_yy
        self.c_xy = c_xy

        # left = A, right = B
        self.logger.info("solving")
        xleft = np.dot(c_xy, np.linalg.solve(c_yy,c_xy.T))
        xright = c_xx
        x_eigvals, x_eigvecs = self.solve_eigprob(xleft, xright)

        yleft = np.dot(c_xy.T, np.linalg.solve(c_xx,c_xy))
        yright = c_yy
        y_eigvals, y_eigvecs = self.solve_eigprob(yleft, yright)

        # y_eigvecs = (1 / np.sqrt(x_eigvals)) * c_yy * c_xy * x_eigvals

        # substitute local variables for member variables
        self.x_weights = x_eigvecs
        self.eigvals = x_eigvals
        self.y_weights = y_eigvecs

    def transform(self, x, y):

        self.logger.info("Normalizing")
        x = self.normalize(x)
        y = self.normalize(y)

        self.logger.info("transform matrices by CCA")
        x_projected = np.dot(x, self.x_weights)
        y_projected = np.dot(y, self.y_weights)

        self.x_c = x_projected
        self.y_c = y_projected

        return x_projected, y_projected

    def ptransform(self, x, y, beta=0.5):

        x_projected, y_projected = self.transform(x, y)

        I = np.eye(len(self.eigvals))
        lamb = np.diag(self.eigvals)
        mat1 = np.linalg.solve(I - np.diag(self.eigvals**2), I)
        mat2 = -np.dot(mat1, lamb)
        mat12 = np.vstack((mat1, mat2))
        mat21 = np.vstack((mat2, mat1))
        mat = np.hstack((mat12, mat21))
        # print lamb.shape, lamb
        p = np.vstack((lamb**beta, lamb**(1-beta)))
        q = np.vstack((x_projected.T, y_projected.T))
        # print p.T.shape, mat.shape, q.shape
        z = np.dot(p.T, np.dot(mat, q)).T[:,:self.n_components]

        self.x_c = x_projected
        self.y_c = y_projected
        self.z_c = z

        return x_projected, y_projected, z

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x, y)

    def fit_ptransform(self, x, y, beta=0.5):
        self.fit(x, y)
        return self.ptransform(x, y, beta)

    def save_params(self, filepath):
        self.logger.info("saving cca")
        np.save(filepath + "n_components.npy" , self.n_components)
        np.save(filepath + "reg_param.npy", self.reg_param)
        np.save(filepath + "x_weights.npy", self.x_weights)
        np.save(filepath + "y_weights.npy", self.y_weights)
        np.save(filepath + "eigvals.npy", self.eigvals)
        np.save(filepath + "cxx.npy", self.c_xx)
        np.save(filepath + "cyy.npy", self.c_yy)
        np.save(filepath + "cxy.npy", self.c_xy)

    def load_params(self, filepath):
        self.logger.info("loading cca")
        self.n_components = np.load(filepath + "n_components.npy")
        self.reg_param = np.load(filepath + "reg_param.npy")
        self.x_weights = np.load(filepath + "x_weights.npy")
        self.y_weights = np.load(filepath + "y_weights.npy")
        self.eigvals = np.load(filepath + "eigvals.npy")
        self.c_xx = np.load(filepath + "cxx.npy")
        self.c_yy = np.load(filepath + "cyy.npy")
        self.c_xy = np.load(filepath + "cxy.npy")

    def check_fit_finished(self):
        return self.x_weights is not None\
               and self.y_weights is not None\
               and self.eigvals is not None\
               and self.x is not None\
               and self.y is not None\
               and self.c_xx is not None\
               and self.c_yy is not None\
               and self.c_xy is not None

    def plot_original_data(self, X, Y):
        plt.subplot(221)
        plt.plot(X[:, 0], X[:, 1], 'xb')
        plt.title('X')

        plt.subplot(222)
        plt.plot(Y[:, 0], Y[:, 1], '.r')
        plt.title('Y')

        plt.show()


    def plot_cca_result(self, probabilistic=True):

        X = None
        Y = None
        Z = None
        if probabilistic:
            self.logger.info("plotting PCCA")
            X = self.x_c
            Y = self.y_c
            Z = self.z_c
        else:
            self.logger.info("plotting CCA")
            X = self.x_c
            Y = self.y_c

        # correct direction
        cor_signs = np.sign([np.corrcoef(X[:, i], Y[:, i])[0, 1] for i in xrange(X.shape[1])])
        Y_s = Y * cor_signs

        # begin plot
        plt.figure()

        plt.subplot(221)
        plt.plot(X[:, 0], X[:, 1], 'xb')
        plt.plot(Y_s[:, 0], Y_s[:, 1], '.r')
        plt.title('CCA XY')

        plt.subplot(222)
        plt.plot(X[:, 0], X[:, 1], 'xb')
        plt.title('CCA X')

        plt.subplot(223)
        plt.plot(Y_s[:, 0], Y_s[:, 1], '.r')
        plt.title('CCA Y')

        if probabilistic:
            plt.subplot(224)
            plt.plot(Z[:, 0], Z[:, 1], 'xb')
            plt.title('CCA Z')

        plt.show()



    def normalize(self, mat):
        m = np.mean(mat, axis=0)
        mat = mat - m
        return mat

    def corrcoef(self):
        return np.corrcoef(self.x_c[:,0], self.y_c[:,0])

    def fix_reverse(self):
        cor = [np.corrcoef(self.x_c[:, i], self.y_c[:, i])[0, 1] for i in xrange(self.x_c.shape[1])]
        cor_signs = np.sign(cor)
        self.y_s = self.y_c * cor_signs

if __name__=="__main__":

    # Reduce dimensions of x, y from 30, 20 to 10 respectively.
    x = np.random.random((100, 30))
    y = np.random.random((100, 20))
    cca = CCA(n_components=10, reg_param=0.1)
    x_c, y_c = cca.fit_transform(x, y)

    #
    print np.corrcoef(x_c[:,0], y_c[:,0])