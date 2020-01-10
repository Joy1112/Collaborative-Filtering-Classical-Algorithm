import os
# import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()

# model
config.model = edict()
config.model.root_path = './models'

# constant
config.N_items = 1682
config.N_users = 943
config.rating_min = 1
config.rating_max = 5
config.epoch_num = 50

# output
config.outputs = edict()
config.outputs.root_path = './outputs'

# feature number
# config.feature_num_list = [2, 4, 8, 16, 32, 64, 128, 256, 512]
config.exp = edict()
config.exp.feature_num_list = [2, 4, 8, 16, 32, 64, 128]
config.exp.dataset_list = ['u1']
config.exp.algo_list = ['svd_bias']

config.svd = edict()
config.svd.gamma = 0.001
config.svd.lamb = 0.1
