#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import numpy as np

from gcca import GCCA
from cca import CCA
import logging
import h5py

class HierarchicalCCA(GCCA):

    def __init__(self, n_components=2, reg_param=0.1):
        GCCA.__init__(self, n_components, reg_param)

        self.cca1 = CCA(self.n_components, self.reg_param)
        self.cca2 = CCA(self.n_components, self.reg_param)

        self.z_list = []

    def fit(self, x0, x1, x2):

        self.data_num = 3

        # 1
        self.cca1.fit(x0, x1)
        self.cca1.transform(x0, x1)

        # 2
        z0 = self.cca1.z_list[0]
        z1 = self.cca1.z_list[1]
        z_all = np.vstack([z0, z1])
        x_dup = np.vstack([x2, x2])
        self.cca2.fit(z_all, x_dup)


    def transform(self, x0, x1, x2):

        # 1
        self.cca1.transform(x0, x1)

        # 2
        z0 = self.cca1.z_list[0]
        z1 = self.cca1.z_list[1]
        z_all = np.vstack([z0, z1])
        x_dup = np.vstack([x2, x2])
        self.cca2.transform(z_all, x_dup)
        w_all, w2_dup = self.cca2.z_list
        w0 = w_all[:z0.shape[0]]
        w1 = w_all[z0.shape[0]:]
        w2 = w2_dup[:x2.shape[0]]

        self.z_list = [w0, w1, w2]

    def save_params(self, filepath):
        self.logger.info("saving hierarchical cca to %s", filepath)
        with h5py.File(filepath, 'w') as f:
            f.create_dataset("n_components", data=self.n_components)
            f.create_dataset("reg_param", data=self.reg_param)
            f.create_dataset("data_num_all", data=self.data_num)
            f.create_dataset("data_num1", data=self.cca1.data_num)
            f.create_dataset("data_num2", data=self.cca2.data_num)

            cov_grp1 = f.create_group("cov_mat1")
            for i, row in enumerate(self.cca1.cov_mat):
                for j, cov in enumerate(row):
                    cov_grp1.create_dataset(str(i) + "_" + str(j), data=cov)

            cov_grp2 = f.create_group("cov_mat2")
            for i, row in enumerate(self.cca2.cov_mat):
                for j, cov in enumerate(row):
                    cov_grp2.create_dataset(str(i) + "_" + str(j), data=cov)

            h_grp1 = f.create_group("h_list1")
            for i, h in enumerate(self.cca1.h_list):
                h_grp1.create_dataset(str(i), data=h)

            h_grp2 = f.create_group("h_list2")
            for i, h in enumerate(self.cca2.h_list):
                h_grp2.create_dataset(str(i), data=h)

            f.create_dataset("eig_vals1", data=self.cca1.eigvals)
            f.create_dataset("eig_vals2", data=self.cca2.eigvals)

            if len(self.cca1.z_list) != 0:
                z_grp1 = f.create_group("z_list1")
                for i, z in enumerate(self.z_list):
                    z_grp1.create_dataset(str(i), data=z)

            if len(self.cca2.z_list) != 0:
                z_grp2 = f.create_group("z_list2")
                for i, z in enumerate(self.cca2.z_list):
                    z_grp2.create_dataset(str(i), data=z)


            if len(self.z_list) != 0:
                z_grp3 = f.create_group("z_list_all")
                for i, z in enumerate(self.z_list):
                    z_grp3.create_dataset(str(i), data=z)

            f.flush()

    def load_params(self, filepath):
        self.logger.info("loading hierarchical cca from %s", filepath)
        with h5py.File(filepath, "r") as f:
            self.n_components = f["n_components"].value
            self.reg_param = f["reg_param"].value
            self.cca1.n_components = self.n_components
            self.cca1.reg_param = self.reg_param
            self.data_num = f["data_num_all"].value
            self.cca1.data_num = f["data_num1"].value
            self.cca2.data_num = f["data_num2"].value

            self.cca1.cov_mat = [[np.array([]) for col in range(self.cca1.data_num)] for row in range(self.cca1.data_num)]
            self.cca2.cov_mat = [[np.array([]) for col in range(self.cca2.data_num)] for row in range(self.cca2.data_num)]

            for i in xrange(self.cca1.data_num):
                for j in xrange(self.cca1.data_num):
                    self.cca1.cov_mat[i][j] = f["cov_mat1/" + str(i) + "_" + str(j)]

            for i in xrange(self.cca2.data_num):
                for j in xrange(self.cca2.data_num):
                    self.cca2.cov_mat[i][j] = f["cov_mat2/" + str(i) + "_" + str(j)]

            self.cca1.h_list = [None] * self.data_num
            for i in xrange(self.cca1.data_num):
                self.cca1.h_list[i] = f["h_list1/" + str(i)].value
            self.cca2.h_list = [None] * self.data_num
            for i in xrange(self.cca2.data_num):
                self.cca2.h_list[i] = f["h_list2/" + str(i)].value
            self.cca1.eig_vals = f["eig_vals1"].value
            self.cca2.eig_vals = f["eig_vals2"].value

            if "z_list1" in f:
                self.cca1.z_list = [None] * self.cca2.data_num
                for i in xrange(self.cca1.data_num):
                    self.cca1.z_list[i] = f["z_list1/" + str(i)].value

            if "z_list2" in f:
                self.cca2.z_list = [None] * self.cca2.data_num
                for i in xrange(self.cca2.data_num):
                    self.cca2.z_list[i] = f["z_list2/" + str(i)].value

            if "z_list_all" in f:
                self.z_list = [None] * self.data_num
                for i in xrange(self.data_num):
                    self.z_list[i] = f["z_list_all/" + str(i)].value

            f.flush()

def main():
    # set log level
    logging.root.setLevel(level=logging.INFO)

    # create data in advance
    a = np.random.rand(50, 50)
    b = np.random.rand(50, 60)
    c = np.random.rand(50, 70)
    d = np.random.rand(50, 80)
    e = np.random.rand(50, 90)
    f = np.random.rand(50, 100)
    g = np.random.rand(50, 110)
    h = np.random.rand(50, 120)
    i = np.random.rand(50, 130)
    j = np.random.rand(50, 140)
    k = np.random.rand(50, 150)

    # create instance of GCCA
    hcca = HierarchicalCCA(reg_param=0.01)
    # calculate GCCA
    hcca.fit(a, b, c)
    # transform
    hcca.transform(a, b, c)
    # save
    hcca.save_params("save/hcca.h5")
    # load
    hcca.load_params("save/hcca.h5")
    # plot
    hcca.plot_result()
    # calc correlations
    hcca.calc_correlations()

if __name__=="__main__":

    main()