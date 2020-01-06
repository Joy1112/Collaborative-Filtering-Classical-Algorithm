import os
import sys
import numpy as np
from cfg.config import config as cfg


def load_data(path):
    data_list = []
    with open(path, 'r') as f:
        data_list = f.readlines()
        f.close()
    n_samples = len(data_list)
    rating_mat = np.zeros([cfg.N_user, cfg.N_item], dtype=np.float32)
    for l in data_list:
        data = l.split('\t')
        rating_mat[int(data[0]), int(data[1])] = float(data[2])
    return n_samples, rating_mat


def find_path(filename, dataset='ml-100k'):
    root_path = os.path.abspath('.')
    root_data_path = os.path.join(root_path, 'data', dataset, filename)
    return root_data_path
