import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def make_batch(batch_size=32):
    z1 = np.random.randint(2, size=batch_size)
    x = np.random.normal(loc=z1*2-1, scale=0.1)
    z2 = np.random.randint(low=z1, high=z1*2+1, size=batch_size)
    # y = np.random.normal(loc=z2, scale=0.1) #stochastic y
    y = z2 + x - z1*2 + 1 #less stochastic y
    return x, y, z1

def tflayersgmm(inputs, n_gauss=8, epsilon=1e-5):
    gmms = tf.layers.dense(
            inputs=inputs,
            units=n_gauss*3)
    tf.reshape(gmms, (-1, n_gauss, 3))
    phi, mu, sigma = tf.split(
        value=gmms,
        axis=-1,
        num_or_size_splits=3)
    phi = tf.nn.softmax(phi, axis=-1)
    sigma = tf.abs(sigma)
    return phi, mu, sigma

def gmm_loss(phi, mu, sigma, x, epsilon=1e-5):
    prob_n = 1/(sigma + epsilon) * \
        tf.exp(-tf.square(x - mu)/(epsilon + 2 * sigma**2))
    prob = tf.reduce_sum(
        tf.multiply(phi, prob_n),
        axis=-1,
        keep_dims=True)
    return prob

def gmm_prob(phi, mu, sigma, x):
    phi = phi[:,None]
    mu = mu[:,None]
    sigma = sigma[:,None]
    x = x[None,:]
    return np.sum(phi * (1/sigma * np.exp(-(x - mu)**2/(2 * sigma **2))), 0)

def simple_mode_finder(phi, mu, sigma,):
    lb = np.min(mu)
    ub = np.max(mu)
    ls = np.linspace(lb, ub)
    return ls[np.argmax(gmm_prob(phi, mu, sigma, ls))]

def sampler(phi, mu, sigma,):
    idx = np.sum(np.random.rand() > np.cumsum(phi) + 1e-5)
    return np.random.normal(loc=mu[idx], scale=sigma[idx])

def run_l2_experiment():
    tf_x = tf.placeholder(
        shape=(None, 1),
        dtype=tf.float32)
    tf_y = tf.placeholder(
        shape=(None, 1),
        dtype=tf.float32)
    
    d1 = tf.layers.dense(
        inputs=tf_x,
        units=32,
        activation=tf.nn.relu)
    
    pred = tf.layers.dense(
    inputs=d1,
    units=1)
    loss = tf.reduce_mean(tf.squared_difference(pred, tf_y))

    opt = tf.train.AdamOptimizer(1e-3)
    train_op = opt.minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            batch_x, batch_y, _ = make_batch()
            _, l = sess.run([train_op, loss],
                      feed_dict={tf_x: batch_x[:,None],
                                  tf_y: batch_y[:,None]})
            if i%100==0:
                print(i, l)
        pred_y = [[], []]
        for i in range(100):
            batch_x, batch_y, batch_z = make_batch(batch_size=10)
            pr = sess.run(pred,
                          feed_dict={tf_x: batch_x[:,None],
                                    tf_y: batch_y[:,None]})
            for p, z in zip(pr, batch_z):
                pred_y[z] += [p[0]]
    return pred_y

def run_mmr_experiment():
    tf_x = tf.placeholder(
        shape=(None, 1),
        dtype=tf.float32)
    tf_y = tf.placeholder(
        shape=(None, 1),
        dtype=tf.float32)
    
    d1 = tf.layers.dense(
        inputs=tf_x,
        units=32,
        activation=tf.nn.relu)
    
    phi, mu, sigma = tflayersgmm(
        inputs=d1,
        n_gauss=8)
    loss_gmm = gmm_loss(
        phi, mu, sigma, tf_y)
    loss = -tf.reduce_mean(tf.log(loss_gmm))

    opt = tf.train.AdamOptimizer(1e-3)
    train_op = opt.minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            batch_x, batch_y, _ = make_batch()
            _, l = sess.run([train_op, loss],
                      feed_dict={tf_x: batch_x[:,None],
                                  tf_y: batch_y[:,None]})
            if i%100==0:
                print(i, l)
        pred_y = [[], []]
        for i in range(100):
            batch_x, batch_y, batch_z = make_batch(batch_size=10)
            ph, m, s = sess.run([phi, mu, sigma],
                          feed_dict={tf_x: batch_x[:,None],
                                    tf_y: batch_y[:,None]})
            pr = [sampler(_p, _m, _s) 
                  for _p, _m, _s in zip(ph, m, s)]
            for p, z in zip(pr, batch_z):
                pred_y[z] += [p]
    return pred_y

def main():
    l2_pred_y = run_l2_experiment()
    mmr_pred_y = run_mmr_experiment()
    x, y, z = make_batch(1000)
    ax=plt.subplot(4,1,1)
    plt.hist(x[z.astype(bool)], 30, lw=0)
    plt.hist(x[~z.astype(bool)], 30, lw=0)
    ax.set_title('Input')
    ax=plt.subplot(4,1,2)
    plt.hist(y[z.astype(bool)], 60, lw=0)
    plt.hist(y[~z.astype(bool)], 30, lw=0)
    ax.set_title('Ground Truth Output')
    ax.set_xlim([-.5, 2.5])
    ax=plt.subplot(4,1,3)
    plt.hist(l2_pred_y[1], 60, lw=0)
    plt.hist(l2_pred_y[0], 30, lw=0)
    ax.set_title('Pred (trained w l2)')
    ax.set_xlim([-.5, 2.5])
    ax=plt.subplot(4,1,4)
    plt.hist(mmr_pred_y[1], 60, lw=0)
    plt.hist(mmr_pred_y[0], 30, lw=0)
    ax.set_title('(Sampled) Pred (trained w mmr)')
    ax.set_xlim([-.5, 2.5])
    plt.tight_layout()

if __name__ == '__main__':
    main()