import os
import sys
sys.path.append(os.getcwd())
import yaml
import torch
import numpy as np

if __name__ == '__main__':
    # manual seed
    config = yaml.load(open(sys.argv[1], 'r'), Loader=yaml.FullLoader)
    # print(os.getcwd())
    print(config.pop('model'))
    # print(config.pop(''))