import os
import sys
import numpy as np
from cfg.config import config as cfg


def loadData(path):
    data_list = []
    with open(path, 'r') as f:
        data_list = f.readlines()
        f.close()
    n_samples = len(data_list)
    rating_mat = np.zeros([cfg.N_users, cfg.N_items], dtype=np.float32)
    for l in data_list:
        data = l.split('\t')
        rating_mat[int(data[0]) - 1, int(data[1]) - 1] = float(data[2])
    return n_samples, rating_mat


def findPath(filename, dataset='ml-100k'):
    root_path = os.path.abspath('.')
    root_data_path = os.path.join(root_path, 'data', dataset, filename)
    return root_data_path


def saveData(filename, data_list):
    with open(filename, 'w') as f:
        for data in data_list:
            f.write(str(data))
            f.write('\n')
        f.close()

if __name__ == '__main__':
    print(cfg)
