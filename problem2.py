import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from nd import run_mmr_experiment, run_gmm_clsfy_experiment, plot3d

def make_batch(ndims, batch_size=32):
    z1 = np.random.randint(1, 10, size=batch_size)
    z1 = np.ceil(z1/4 - 1).astype(int)
    x = np.random.normal(loc=z1*2-1, scale=0.1)
    z2 = np.random.randint(low=z1, high=z1*2+1, size=batch_size)
    # y = np.random.normal(loc=z2, scale=0.1) #stochastic y
    y = z2 + x - z1*2 + 1#less stochastic y
    y = (z1 == 2) + (z1 != 2) * y
    scale = np.stack([.1 - 0.09 * (z1==2)] * ndims, -1)
    nd_y = np.random.normal(np.stack([y] * ndims, -1), scale=scale)
    return x, nd_y, z1

if __name__ == '__main__':
    ndims = 2
    np.random.seed(0)
    tf.random.set_random_seed(0)
    mmr_pred_y = run_mmr_experiment(make_batch, ndims)
    gmm_clsfy_y = run_gmm_clsfy_experiment(make_batch, ndims, n_gauss=4)
    
    x, y, z = make_batch(ndims, batch_size=1000)   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot3d(ax, y[z==1, 0], y[z==1, 1], vis_bins=31)
    plot3d(ax, y[z==0, 0], y[z==0, 1], vis_bins=31)
    plot3d(ax, y[z==2, 0], y[z==2, 1], vis_bins=31)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot3d(ax, *mmr_pred_y[1][:2], vis_bins=31)
    plot3d(ax, *mmr_pred_y[0][:2], vis_bins=31)
    plot3d(ax, *mmr_pred_y[2][:2], vis_bins=31)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot3d(ax, *gmm_clsfy_y[1][:2], vis_bins=31)
    plot3d(ax, *gmm_clsfy_y[0][:2], vis_bins=31)
    plot3d(ax, *gmm_clsfy_y[2][:2], vis_bins=31)