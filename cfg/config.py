import os
import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()

# model
config.model = edict()
config.model.root_path = './models'

# constant
config.N_item = 1682
config.N_user = 943
config.rating_min = 1
config.rating_max = 5

# output
config.outputs = edict()
config.outputs.root_path = './outputs'
