import math
import time
import copy
import pickle
import numpy as np
from cfg.config import config as cfg
from lib.utils.create_logger import print_and_log


class SVD(object):
    """
    Implementation of SVD for Collaborative Filtering.
    Reference:
        Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, (8), 30-37.
    """
    def __init__(self, feature_num, gamma, lamb, epoch_num, logger, loss_threshold=None, save_model=False):
        self.feature_num = feature_num
        self.gamma = gamma
        self.lamb = lamb
        self.epoch_num = epoch_num
        self.loss_threshold = loss_threshold
        self.logger = logger
        self.save_model = save_model

        self.n_users = cfg.N_users
        self.n_items = cfg.N_items

    def trainSGD(self, train_data, eval_data=None):
        """
        Training the model with SGD.
        rating_matrix = user_matrix * (item_matrix)^{T}
        args:
            train_data:
                the training data matrix, only sparse data.
                shape: [n_users, n_items]
                type: np.ndarray(np.int32)
            eval_data:
                the evaluation data matrix.
                shape: [n_users, n_items]
                type: np.ndarray(np.int32)
        """
        train_rmse_loss_list = []
        valid_rmse_loss_list = []
        accuracy_list = []

        # number of train samples
        train_sample_num = train_data[np.nonzero(train_data)].shape[0]

        # p, q correspond to user_matrix and item_matrix
        p_mat = np.random.random_sample([self.n_users, self.feature_num])
        q_mat = np.random.random_sample([self.n_items, self.feature_num])
        p_mat_new = copy.deepcopy(p_mat)
        q_mat_new = copy.deepcopy(q_mat)

        start = time.time()
        for epoch in range(self.epoch_num):
            train_rmse_loss = 0.0
            for u in range(self.n_users):
                for i in range(self.n_items):
                    if train_data[u, i] != 0:
                        e_ui = train_data[u, i] - np.dot(q_mat[i, :].T, p_mat[u, :])
                        train_rmse_loss += e_ui**2
                        p_mat_new[u, :] += self.gamma * (e_ui * q_mat[i, :] - self.lamb * p_mat[u, :])
                        q_mat_new[i, :] += self.gamma * (e_ui * p_mat[u, :] - self.lamb * q_mat[i, :])
                        p_mat[u, :] = copy.deepcopy(p_mat_new[u, :])
                        q_mat[i, :] = copy.deepcopy(q_mat_new[i, :])
            end = time.time()
            train_rmse_loss = np.sqrt(train_rmse_loss / train_sample_num)

            if eval_data.any():
                valid_rmse_loss, accuracy = self.evaluate(train_data, eval_data, p_mat, q_mat)
                train_rmse_loss_list.append(train_rmse_loss)
                valid_rmse_loss_list.append(valid_rmse_loss)
                accuracy_list.append(accuracy)

                print_and_log("Epoch {:d}: totally training time {:.4f}, training RMSE: {:.4f}, validation RMSE: {:.4f}, accuracy: {:.4f}%"
                              .format(epoch, end - start, train_rmse_loss, valid_rmse_loss, accuracy * 100), self.logger)
                # print("Epoch {:d}: totally training time {:.4f}, training RMSE: {:.4f}, validation RMSE: {:.4f}, accuracy: {:.4f}%"
                #       .format(epoch, end - start, train_rmse_loss, valid_rmse_loss, accuracy * 100))
            else:
                print_and_log("Epoch {:d}: totally training time {:.4f}, training RMSE: {:.4f}"
                              .format(epoch, end - start, train_rmse_loss), self.logger)

            # when the train_rmse_loss is less than the threshold, end the training process.
            if self.loss_threshold is not None:
                if train_rmse_loss < self.loss_threshold:
                    break

        return train_rmse_loss_list, valid_rmse_loss_list, accuracy_list

    def trainSGDWithBias(self, train_data, eval_data=None):
        """
        Training the model with SGD, the model has the bias.
        rating_matrix = user_matrix * (item_matrix)^{T} + b, where b_ui = \mu + b_{i} + b_{u}
        args:
            train_data:
                the training data matrix, only sparse data.
                shape: [n_users, n_items]
                type: np.ndarray(np.int32)
            eval_data:
                the evaluation data matrix.
                shape: [n_users, n_items]
                type: np.ndarray(np.int32)
        """
        # mean of all the non-zero elements
        mu = np.mean(train_data[np.nonzero(train_data)])
        # number of train samples
        train_sample_num = train_data[np.nonzero(train_data)].shape[0]

        # p, q correspond to user_matrix and item_matrix
        p_mat = np.random.random_sample([self.n_users, self.feature_num])
        q_mat = np.random.random_sample([self.n_items, self.feature_num])
        p_mat_new = p_mat
        q_mat_new = q_mat

        b_u = np.ones([self.n_users])
        b_i = np.ones([self.n_items])

        start = time.time()
        for epoch in range(self.epoch_num):
            train_rmse_loss = 0.0
            for u in range(self.n_users):
                for i in range(self.n_items):
                    if train_data[u, i] != 0:
                        b_ui = b_u[u] + b_i[i] + mu
                        e_ui = train_data[u, i] - np.dot(q_mat.T, p_mat) - b_ui
                        train_rmse_loss += e_ui**2
                        p_mat_new[u, :] += self.gamma * (e_ui * q_mat[i, :] - self.lamb * p_mat[u, :])
                        q_mat_new[i, :] += self.gamma * (e_ui * p_mat[u, :] - self.lamb * q_mat[i, :])
                        p_mat[u, :] = p_mat_new[u, :]
                        q_mat[i, :] = q_mat_new[i, :]
                        b_u[u] += self.gamma * (e_ui - self.lamb * b_u[u])
                        b_i[i] += self.gamma * (e_ui - self.lamb * b_i[i])
            end = time.time()
            train_rmse_loss = np.sqrt(train_rmse_loss / train_sample_num)

            if eval_data.any():
                valid_rmse_loss, accuracy = self.evaluate(train_data, eval_data, p_mat, q_mat, b_u, b_i, mu)
                train_rmse_loss_list.append(train_rmse_loss)
                valid_rmse_loss_list.append(valid_rmse_loss)
                accuracy_list.append(accuracy)
                print_and_log("Epoch {:d}: totally training time {:.4f}, training RMSE: {:.4f}, validation RMSE: {:.4f}, accuracy: {:.4f}%"
                              .format(epoch, end - start, train_rmse_loss, valid_rmse_loss, accuracy * 100), self.logger)
            else:
                print_and_log("Epoch {:d}: totally training time {:.4f}, training RMSE: {:.4f}"
                              .format(epoch, end - start, train_rmse_loss), self.logger)

            # when the train_rmse_loss is less than the threshold, end the training process.
            if self.loss_threshold is not None:
                if train_rmse_loss < self.loss_threshold:
                    break

        return train_rmse_loss_list, valid_rmse_loss_list, accuracy_list

    def evaluate(self, train_data, eval_data, p_mat, q_mat, b_u=None, b_i=None, mu=None):
        """
        Evaluate the model with the given evaluation data. Here the training data is used to correct the prediction.
        args:
            train_data:
                the training data matrix, only sparse data.
                shape: [n_users, n_items]
                type: np.ndarray(np.int32)
            eval_data:
                the evaluation data matrix.
                shape: [n_users, n_items]
                type: np.ndarray(np.int32)
            p_mat:
                the user matrix.
                shape: [n_users, n_features]
                type: np.ndarray(np.float32)
            q_mat:
                the item matrix.
                shape: [n_items, n_features]
                type: np.ndarray(np.float32)
            b_u:
                the bias of users.
                shape: [n_users, ]
                type: np.ndarray(np.float32)
            b_i:
                the bias of items.
                shape: [n_items, ]
                type: np.ndarray(np.float32)
            mu:
                the mean of the training data.
                type: np.float64
        returns:
            valid_rmse_loss:
                the RMSE loss of the model on the evaluation data.
                type:np.float64
            accuracy:
                the accuracy of the model on the evaluation data.
                type: np.float64
        """
        eval_samples = eval_data[np.nonzero(eval_data)]

        # predict
        if (b_u is None) or (b_i is None) or (mu is None):
            pred = self.predict(train_data, p_mat, q_mat)
        else:
            pred = self.predictWithBias(train_data, p_mat, q_mat, b_u, b_i, mu)
        pred_samples = pred[np.nonzero(eval_data)]

        # compute the mse loss
        valid_rmse_loss = np.sqrt(np.mean((eval_samples - pred_samples)**2))
        accuracy = np.sum(eval_samples == pred_samples) / eval_samples.shape[0]

        return valid_rmse_loss, accuracy

    def predict(self, train_data, p_mat, q_mat):
        """
        Predict the whole rating matrix. Here the training data is used to correct the prediction.
        args:
            train_data:
                the training data matrix, only sparse data.
                shape: [n_users, n_items]
                type: np.ndarray(np.int32)
            p_mat:
                the user matrix.
                shape: [n_users, n_features]
                type: np.ndarray(np.float32)
            q_mat:
                the item matrix.
                shape: [n_items, n_features]
                type: np.ndarray(np.float32)
        returns:
            pred:
                the prediction of the whole rating matrix.
                shape: [n_users, n_items]
                type: np.ndarray(np.int32)
        """
        # obtain the prediction
        pred = np.dot(p_mat, q_mat.T)

        # fit the prediction to an int variable in [1, 5]
        pred = np.around(pred).astype(np.int32)
        pred = np.clip(pred, 1, 5)
        # for the element already in train_data, set them to the train sample
        fliter_pred = (train_data == 0)
        pred = np.multiply(fliter_pred, pred) + train_data

        return pred

    def predictWithBias(self, train_data, p_mat, q_mat, b_u, b_i, mu):
        """
        Predict the whole rating matrix with the bias model. Here the training data is used to correct the prediction.
        args:
            train_data:
                the training data matrix, only sparse data.
                shape: [n_users, n_items]
                type: np.ndarray(np.int32)
            p_mat:
                the user matrix.
                shape: [n_users, n_features]
                type: np.ndarray(np.float32)
            q_mat:
                the item matrix.
                shape: [n_items, n_features]
                type: np.ndarray(np.float32)
            b_u:
                the bias of users.
                shape: [n_users, ]
                type: np.ndarray(np.float32)
            b_i:
                the bias of items.
                shape: [n_items, ]
                type: np.ndarray(np.float32)
            mu:
                the mean of the training data.
                type: np.float64
        returns:
            pred:
                the prediction of the whole rating matrix.
                shape: [n_users, n_items]
                type: np.ndarray(np.int32)
        """
        # obtain the prediction
        pred = np.dot(p_mat, q_mat.T)
        for u in range(self.n_users):
            for i in range(self.n_items):
                pred += mu + b_u[u] + b_i[i]

        # fit the prediction to an int variable in [1, 5]
        pred = np.around(pred).astype(np.int32)
        pred = np.clip(pred, 1, 5)
        # for the element already in train_data, set them to the train sample
        fliter_pred = (train_data == 0)
        pred = np.multiply(fliter_pred, pred) + train_data

        return pred
