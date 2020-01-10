import os
import math
import time
import pickle
import numpy as np

from cfg.config import config as cfg
from lib.utils.data_util import loadData, findPath, saveData
from lib.utils.create_logger import create_logger, print_and_log
from lib.classic_algo import svd, nmf


def main():
    for dataset in cfg.exp.dataset_list:
        for algo in cfg.exp.algo_list:
            logger, final_output_path = create_logger(cfg.outputs.root_path, algo, dataset)
            cfg.outputs.final_output_path = final_output_path

            print_and_log("The dataset is {}".format(dataset), logger)
            print_and_log("The algorithm is {}".format(algo), logger)

            train_data_path = findPath(str(dataset) + '.base')
            eval_data_path = findPath(str(dataset) + '.test')

            _, train_data = loadData(train_data_path)
            _, eval_data = loadData(eval_data_path)

            for feature_num in cfg.exp.feature_num_list:
                print_and_log('The feature number is {}'.format(feature_num), logger)
                if algo == 'svd':
                    model = svd.SVD(feature_num, cfg.svd.gamma, cfg.svd.lamb, cfg.epoch_num, logger=logger)
                    train_rmse, valid_rmse, accuracy = model.trainSGD(train_data, eval_data)
                elif algo == 'svd_bias':
                    model = svd.SVD(feature_num, cfg.svd.gamma, cfg.svd.lamb, cfg.epoch_num, logger=logger)
                    train_rmse, valid_rmse, accuracy = model.trainSGDWithBias(train_data, eval_data)
                elif algo == 'nmf':
                    model = nmf.WNMF(feature_num, cfg.epoch_num, logger=logger)
                    train_rmse, valid_rmse, accuracy = model.train(train_data, eval_data)

                file_prefix = os.path.join(final_output_path, 'dataset_' + str(dataset) + '_' + str(algo) + '_feature_' + str(feature_num))
                saveData(file_prefix +'_train_rmse.txt', train_rmse)
                saveData(file_prefix +'_valid_rmse.txt', valid_rmse)
                saveData(file_prefix +'_accuracy.txt', accuracy)



if __name__ == '__main__':
    main()
