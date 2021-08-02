import os
import numpy as np
import glob
import gc
from multiprocessing import Pool

path = '/home/asap7772/asap7772/real_data_kitchen/bridge_data_numpy_shifted/toykitchen1/'

dirs = glob.glob(path + '*/out.npy')

def generate_rew_npy(x):
    print(x)
    data = np.load(x, allow_pickle=True)
    rews = []
    for i in range(len(data)):
        data[i]['rewards'] = data[i]['terminals'][:]
        labeled_rew = data[i]['terminals'][:]
        labeled_rew[-2:] = [1,1]
        rews.append(labeled_rew)
    np.save(x, data)
    split_path = x.split('.')
    np.save(split_path[0]+'_rew.npy', rews)

if __name__ == "__main__":
    with Pool(16) as pool:
        pool.map(generate_rew_npy, dirs)
