import math
import time
import numpy as np
from cfg.config import config as cfg


class SVD(object):
    """
    Implementation of SVD for Collaborative Filtering.
    Reference:
        Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, (8), 30-37.
    """
    def __init__(self, data_mat, feature_num, gamma, lamb, epoch_num):
        self.data_mat = data_mat
        self.feature_num = feature_num
        self.gamma = gamma
        self.lamb = lamb
        self.epoch_num = epoch_num

        self.n_users = cfg.N_users
        self.n_items = cfg.N_items

    def train_sgd(self):
        # p, q correspond to user_matrix and item_matrix
        p_mat = np.random.random_sample([self.n_users, self.feature_num])
        q_mat = np.random.random_sample([self.n_items, self.feature_num])

        start = time.time()
        for epoch in range(self.epoch_num):
            for u in range(self.n_users):
                for i in range(self.n_items):
                    if self.data_mat[u, i] != 0:
                        e_ui = self.data_mat[u, i] - np.dot(q_mat.T, p_mat)
                        p_mat[u, :] += self.gamma * (e_ui * q_mat[i, :] - self.lamb * p_mat[u, :])
                        q_mat[i, :] += self.gamma * (e_ui * p_mat[u, :] - self.lamb * q_mat[i, :])

        return p_mat, q_mat

    def train_sgd_with_bias(self):
        # mean of all the non-zero elements
        mu = np.mean(self.data_mat[np.nonzero(self.data_mat)])

        # p, q correspond to user_matrix and item_matrix
        p_mat = np.random.random_sample([self.n_users, self.feature_num])
        q_mat = np.random.random_sample([self.n_items, self.feature_num])

        b_u = np.ones([self.n_users])
        b_i = np.ones([self.n_items])
        
        start = time.time()
        for epoch in range(self.epoch_num):
            for u in range(self.n_users):
                for i in range(self.n_items):
                    if self.data_mat[u, i] != 0:
                        b_ui = b_u[u] + b_i[i] + mu
                        e_ui = self.data_mat[u, i] - np.dot(q_mat.T, p_mat) - b_ui
                        p_mat[u, :] += self.gamma * (e_ui * q_mat[i, :] - self.lamb * p_mat[u, :])
                        q_mat[i, :] += self.gamma * (e_ui * p_mat[u, :] - self.lamb * q_mat[i, :])
                        b_u[u] += self.gamma * (e_ui - self.lamb * b_u[u])
                        b_i[i] += self.gamma * (e_ui - self.lamb * b_i[i])

        return p_mat, q_mat, b_u, b_i
