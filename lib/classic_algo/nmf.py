import math
import time
import copy
import pickle
import numpy as np
from cfg.config import config as cfg
from lib.utils.create_logger import print_and_log


class WNMF(object):
    """
    Implement of the weighted NMF for Collaborative Filtering which is proposed by Zhang, et al.[see Reference].
    Reference:
        Zhang, S., Wang, W., Ford, J., & Makedon, F. (2006, April). Learning from incomplete ratings using non-negative matrix factorization.
        In Proceedings of the 2006 SIAM international conference on data mining (pp. 549-553). Society for Industrial and Applied Mathematics.
    """
    def __init__(self, feature_num, epoch_num, logger, loss_threshold=None, save_model=False):
        self.feature_num = feature_num
        self.epoch_num = epoch_num
        self.logger = logger
        self.loss_threshold = loss_threshold
        self.save_model = save_model

        self.n_users = cfg.N_users
        self.n_items = cfg.N_items

    def train(self, train_data, eval_data=None):
        """
        Training the model with SGD.
        rating_matrix = U_matrix * V_matrix, where U_matrix & V_matrix are non-negative matrix.
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

        # train_samples is the array of the non-zero samples.
        train_samples = train_data[np.nonzero(train_data)]
        # W_mat is the weight matrix, W[i, j] = 1 if train_data[i, j] > 0 else 0.
        W_mat = (train_data > 0).astype(np.int32)

        U_mat = np.random.random_sample([self.n_users, self.feature_num])
        V_mat = np.random.random_sample([self.feature_num, self.n_items])
        U_mat_new = copy.deepcopy(U_mat)
        V_mat_new = copy.deepcopy(V_mat)

        start = time.time()
        for epoch in range(self.epoch_num):
            # obtain the prediction matrix
            A_mat = self.predict(U_mat, V_mat)
            # compute the RMSE loss
            pred_samples = A_mat[np.nonzero(train_data)]
            train_rmse_loss = np.sqrt(np.mean((pred_samples - train_samples)**2))

            # when the train_rmse_loss is less than the threshold, end the training process.
            if self.loss_threshold is not None:
                if train_rmse_loss < self.loss_threshold:
                    break

            # update the model by the 'multiplicative update rules' in WNMF
            U_up_factor = np.dot(np.multiply(W_mat, train_data), V_mat.T)
            U_down_factor = np.dot(np.multiply(W_mat, A_mat), V_mat.T)
            # U_mat_new = np.divide(np.dot(np.multiply(W_mat, train_data), V_mat.T), np.dot(np.multiply(W_mat, A_mat), V_mat.T))
            for i in range(len(np.nonzero(U_up_factor)[0])):
                row = np.nonzero(U_up_factor)[0]
                col = np.nonzero(U_up_factor)[1]
                U_mat_new[row[i], col[i]] = U_up_factor[row[i], col[i]] / U_down_factor[row[i], col[i]]
            U_mat_new = np.multiply(U_mat, U_mat_new)

            # V_mat_new = np.divide(np.dot(U_mat.T, np.multiply(W_mat, train_data)), np.dot(U_mat.T, np.multiply(W_mat, A_mat)))
            V_up_factor = np.dot(U_mat.T, np.multiply(W_mat, train_data))
            V_down_factor = np.dot(U_mat.T, np.multiply(W_mat, A_mat))
            for i in range(len(np.nonzero(V_up_factor)[0])):
                row = np.nonzero(V_up_factor)[0]
                col = np.nonzero(V_up_factor)[1]
                V_mat_new[row[i], col[i]] = V_up_factor[row[i], col[i]] / V_down_factor[row[i], col[i]]
            V_mat_new = np.multiply(V_mat, V_mat_new)
            U_mat = copy.deepcopy(U_mat_new)
            V_mat = copy.deepcopy(V_mat_new)

            end = time.time()

            if eval_data.any():
                valid_rmse_loss, accuracy = self.evaluate(train_data, eval_data, U_mat, V_mat)
                train_rmse_loss_list.append(train_rmse_loss)
                valid_rmse_loss_list.append(valid_rmse_loss)
                accuracy_list.append(accuracy)

                print_and_log("Epoch {:d}: totally training time {:.4f}, training RMSE: {:.4f}, validation RMSE: {:.4f}, accuracy: {:.4f}%"
                              .format(epoch, end - start, train_rmse_loss, valid_rmse_loss, accuracy * 100), self.logger)
            else:
                print_and_log("Epoch {:d}: totally training time {:.4f}, training RMSE: {:.4f}"
                              .format(epoch, end - start, train_rmse_loss), self.logger)

        return train_rmse_loss_list, valid_rmse_loss_list, accuracy_list

    def evaluate(self, train_data, eval_data, U_mat, V_mat):
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
            U_mat:
                the matrix which reveal the relationship between N users and K features. Notice that U_mat is non-negative.
                shape: [n_users, n_features]
                type: np.ndarray(np.float32)
            V_mat:
                the matrix which reveal the relationship between K features and M items. Notice that V_mat is non-negative.
                shape: [n_features, n_items]
                type: np.ndarray(np.float32)
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
        pred = self.predict(U_mat, V_mat, train_data, correct_with_train_data=True)
        pred_samples = pred[np.nonzero(eval_data)]

        # compute the RMSE loss
        valid_rmse_loss = np.sqrt(np.mean((eval_samples - pred_samples)**2))
        accuracy = np.sum(eval_samples == pred_samples) / eval_samples.shape[0]

        return valid_rmse_loss, accuracy

    def predict(self, U_mat, V_mat, train_data=None, correct_with_train_data=False):
        """
        Predict the whole rating matrix. Here the training data is used to correct the prediction.
        args:
            U_mat:
                the matrix which reveal the relationship between N users and K features. Notice that U_mat is non-negative.
                shape: [n_users, n_features]
                type: np.ndarray(np.float32)
            V_mat:
                the matrix which reveal the relationship between K features and M items. Notice that V_mat is non-negative.
                shape: [n_features, n_items]
                type: np.ndarray(np.float32)
            train_data:
                the training data matrix, only sparse data.
                shape: [n_users, n_items]
                type: np.ndarray(np.int32)
            correct_with_train_data:
                whether use the training data to correct the prediction.
                type: Boolean
        returns:
            pred:
                the prediction of the whole rating matrix.
                shape: [n_users, n_items]
                type: np.ndarray(np.int32)
        """
        pred = np.dot(U_mat, V_mat)
        # fit the prediction to an int variable in [1, 5]
        pred = np.around(pred).astype(np.int32)
        pred = np.clip(pred, 1, 5)

        if correct_with_train_data:
            # for the element already in train_data, set them to the train sample
            assert train_data is not None
            fliter_pred = (train_data == 0)
            pred = np.multiply(fliter_pred, pred) + train_data

        return pred
