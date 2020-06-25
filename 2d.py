import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def make_batch(nd, batch_size=32):
    z1 = np.random.randint(2, size=batch_size)
    x = np.random.normal(loc=z1*2-1, scale=0.1)
    z2 = np.random.randint(low=z1, high=z1*2+1, size=batch_size)
    # y = np.random.normal(loc=z2, scale=0.1) #stochastic y
    y = z2 + x - z1*2 + 1 #less stochastic y
    nd_y = np.random.normal(np.stack([y] * nd, -1), scale=.1)
    return x, nd_y, z1

def plot3d(ax, x1, x2, vis_bins=30):
    vis_range = (-.5, 2.5)
    bin_w = (vis_range[1] - vis_range[0])/vis_bins
    x1_data, x2_data = np.meshgrid( np.linspace(*vis_range, vis_bins),
                                  np.linspace(*vis_range, vis_bins) )
    x1_data = x1_data.flatten()
    x2_data = x2_data.flatten()
    
    data_array, _, _ = np.histogram2d(x1,x2,bins=vis_bins, range=[vis_range, vis_range])
    y_data = data_array.flatten()
    y_bar = y_data != 0
    ax.bar3d( x1_data[y_bar],
              x2_data[y_bar],
              np.zeros(len(y_data))[y_bar],
              bin_w, bin_w, y_data[y_bar] )
    ax.set_xlim(*vis_range)
    ax.set_ylim(*vis_range)
    ax.set_xlabel('Height')
    ax.set_ylabel('Weight')

def tflayersgmm(inputs, n_gauss=8, epsilon=1e-5):
    gmms = tf.layers.dense(
            inputs=inputs,
            units=n_gauss*6)
    gmms = tf.reshape(gmms, (-1, n_gauss, 3))
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

def make_sect_and_center(n_bins, overlap, span):
    section_length = (span[1] - span[0])/(n_bins - (n_bins - 1) * overlap)
    sections = [span[0] + i * (1 - overlap) * section_length for i in range(n_bins)]
    sections = [[i, i+section_length] for i in sections]
    centers = [(i+j)/2 for i,j in sections]
    sections[0][0] = -np.inf
    sections[-1][1] = np.inf
    return sections, centers

def mmr_sampler(phi, mu, sigma_inv,):
    idx = np.sum(np.random.rand() > np.cumsum(phi) + 1e-5)
    return np.random.multivariate_normal(
        mean=mu[idx],
        cov=np.linalg.inv(sigma_inv[idx]))

def ind_mb_sampler(cls_probs, reg_value, centers, top_k=3):
    idxs = []
    for d_sort_idx, d_p_vals in zip(np.argsort(cls_probs), cls_probs):
        d_sort_idx = d_sort_idx[:-top_k-1:-1]
        pvals = d_p_vals[d_sort_idx]
        pvals /= np.sum(pvals)
        idx = d_sort_idx[np.random.multinomial(1, pvals).argmax()]
        idxs += [idx]
    return [centers[idx] + r[idx] for idx,r in zip(idxs, reg_value)]

def joint_mb_sampler(cls_probs, reg_value, centers, top_k=3):
    d_sort_idx = np.argsort(cls_probs)[:-top_k-1:-1]
    pvals = cls_probs[d_sort_idx]
    pvals /= np.sum(pvals)
    idx = d_sort_idx[np.random.multinomial(1, pvals).argmax()]
    return centers[:,idx] + reg_value[:,idx]

def run_l2_experiment(nd, train_steps=1000):
    tf_x = tf.placeholder(
        shape=(None, 1),
        dtype=tf.float32)
    tf_y = tf.placeholder(
        shape=(None, nd),
        dtype=tf.float32)
    
    d1 = tf.layers.dense(
        inputs=tf_x,
        units=32,
        activation=tf.nn.relu)
    
    pred = tf.layers.dense(
    inputs=d1,
    units=nd)
    loss = tf.reduce_mean(tf.squared_difference(pred, tf_y))

    opt = tf.train.AdamOptimizer(1e-3)
    train_op = opt.minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(train_steps):
            batch_x, batch_y, _ = make_batch(nd)
            _, l = sess.run([train_op, loss],
                      feed_dict={tf_x: batch_x[:,None],
                                  tf_y: batch_y})
            if i%100==0:
                print(i, l)
        pred_y = [[], []]
        for i in range(100):
            batch_x, batch_y, batch_z = make_batch(nd, batch_size=10)
            pr = sess.run(pred,
                          feed_dict={tf_x: batch_x[:,None],
                                    tf_y: batch_y})
            for p, z in zip(pr, batch_z):
                pred_y[z] += [p]
    pred_y = [np.stack(i, -1) for i in pred_y]
    return pred_y

def run_mmr_experiment(nd, train_steps=1000):
    def make_sigma_inv(x):
        sigma_lt = tf.scatter_nd(np.array(np.tril_indices(nd)).T, x, (nd, nd))
        sigma_lt = tf.where(tf.eye(nd, dtype=tf.bool), tf.exp(sigma_lt), sigma_lt)
        sigma_inv = tf.matmul(sigma_lt, sigma_lt, transpose_b=True)
        return sigma_inv
    tf_x = tf.placeholder(
        shape=(None, 1),
        dtype=tf.float32)
    tf_y = tf.placeholder(
        shape=(None, nd),
        dtype=tf.float32)
    
    d1 = tf.layers.dense(
        inputs=tf_x,
        units=32,
        activation=tf.nn.relu)
    
    n_gauss = 4
    n_mparams = (nd * (nd + 1))//2
    n_gparams = 1 + nd + n_mparams
    gmms = tf.layers.dense(
            inputs=d1,
            units=n_gauss*n_gparams)
    gmms = tf.reshape(gmms, (-1, n_gauss, n_gparams))
    phi = gmms[:,:,0]
    mu = gmms[:,:,1:1+nd]
    sigma_p = gmms[:,:,1+nd:]
    
    phi = tf.nn.softmax(phi, axis=-1)
    sigma_inv = tf.map_fn(make_sigma_inv,
                          tf.reshape(sigma_p, (-1, n_mparams)))
    sigma_inv = tf.reshape(sigma_inv, (-1, n_gauss, nd, nd))
    
    x_minus_mu = tf_y[:,None] - mu
    exponent =  tf.reduce_sum(
        tf.multiply(
            x_minus_mu,
            tf.linalg.matvec(sigma_inv, x_minus_mu)),
        axis=-1)
    prob_n = ((2 * np.pi)**nd * tf.linalg.det(sigma_inv))**.5 * \
        tf.exp(-0.5 * exponent)
    prob = tf.reduce_sum(
        tf.multiply(phi, prob_n),
        axis=-1,
        keep_dims=True)
    sigma_reg = tf.reduce_mean(tf.square(sigma_inv))
    loss = -tf.reduce_mean(tf.log(prob)) + 1e-3 * sigma_reg

    opt = tf.train.AdamOptimizer(1e-3)
    train_op = opt.minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(train_steps):
            batch_x, batch_y, _ = make_batch(nd)
            _, l, si, p, xmm = sess.run([train_op, loss, sigma_inv, prob, x_minus_mu],
                      feed_dict={tf_x: batch_x[:,None],
                                  tf_y: batch_y})
            if i%100==0:
                print(i, l)
        pred_y = [[], []]
        for i in range(100):
            batch_x, batch_y, batch_z = make_batch(nd, batch_size=10)
            ph, m, si = sess.run([phi, mu, sigma_inv],
                          feed_dict={tf_x: batch_x[:,None],
                                    tf_y: batch_y})
            pr = [mmr_sampler(_p, _m, _s) 
                  for _p, _m, _s in zip(ph, m, si)]
            for p, z in zip(pr, batch_z):
                pred_y[z] += [p]
    pred_y = [np.stack(i, -1) for i in pred_y]
    return pred_y

def run_mb_ind_experiment(nd, n_bins=20, train_steps=1000):
    tf_x = tf.placeholder(
        shape=(None, 1),
        dtype=tf.float32)
    tf_y = tf.placeholder(
        shape=(None, nd),
        dtype=tf.float32)
    
    d1 = tf.layers.dense(
        inputs=tf_x,
        units=32,
        activation=tf.nn.relu)
    
    sections, centers = make_sect_and_center(n_bins=n_bins, overlap=0.25, span=(-.5, 2.5))
    tfsections = tf.constant(sections) #20 x nd
    tfcenters = tf.constant(centers) #20
    cls_logit = tf.layers.dense(d1, units=n_bins*nd)
    reg_val = tf.layers.dense(d1, units=n_bins*nd)
    
    cls_logit = tf.reshape(cls_logit, (-1, nd, n_bins))
    cls_probs = tf.nn.softmax(cls_logit)
    reg_val = tf.reshape(reg_val, (-1, nd, n_bins))
    
    best_class = tf.argmin(tf.abs(tfcenters[None,None] - tf_y[:,:,None]), axis=-1)
    cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=best_class,
            logits=cls_logit)
    bucket = tf.logical_and(
            tfsections[None,None,:,0] < tf_y[:,:,None],
            tfsections[None,None,:,1] > tf_y[:,:,None],)
    reg_loss = tf.multiply(
            tf.squared_difference(reg_val + tfcenters, tf_y[:,:,None]),
            tf.cast(bucket, tf.float32))
    loss = tf.reduce_mean(cls_loss) + tf.reduce_mean(reg_loss)

    opt = tf.train.AdamOptimizer(1e-3)
    train_op = opt.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(train_steps):
            batch_x, batch_y, _ = make_batch(nd)
            _, l = sess.run([train_op, loss],
                      feed_dict={tf_x: batch_x[:,None],
                                  tf_y: batch_y})
            if i%100==0:
                print(i, l)
        pred_y = [[], []]
        for i in range(100):
            batch_x, batch_y, batch_z = make_batch(nd, batch_size=10)
            c, r = sess.run([cls_probs, reg_val],
                          feed_dict={tf_x: batch_x[:,None],
                                    tf_y: batch_y})
            pr = [ind_mb_sampler(_c, _r, centers) 
                  for _c, _r in zip(c, r)]
            for p, z in zip(pr, batch_z):
                pred_y[z] += [p]
    pred_y = [np.stack(i, -1) for i in pred_y]
    return pred_y

def run_mb_joint_experiment(nd, n_bins=20, train_steps=1000):
    tf_x = tf.placeholder(
        shape=(None, 1),
        dtype=tf.float32)
    tf_y = tf.placeholder(
        shape=(None, nd),
        dtype=tf.float32)
    
    d1 = tf.layers.dense(
        inputs=tf_x,
        units=32,
        activation=tf.nn.relu)
    
    sections, centers = make_sect_and_center(n_bins=n_bins, overlap=0.25, span=(-.5, 2.5))
    centers = np.stack(np.meshgrid(*[centers] * nd), 0).astype(np.float32).reshape(nd, -1)
    #before reshape - centers: nd x 20 x 20
    #sections: 20 x 2 (lower_bound, upper_bound)
    sections = np.stack([np.meshgrid(*[[i[j] for i in sections]] * nd) 
                         for j in range(2)], 0).astype(np.float32).reshape(2, nd, -1)
    #before reshape - sections: 2[bound] x nd[coordinate] x 20 x 20
    tfsections = tf.constant(sections)
    tfcenters = tf.constant(centers)
    cls_logit = tf.layers.dense(d1, units=n_bins**nd)
    reg_val = tf.layers.dense(d1, units=n_bins**nd*nd)
    cls_probs = tf.nn.softmax(cls_logit)
    
    # cls_logit = tf.reshape(cls_logit, (-1, n_bins, n_bins))
    reg_val = tf.reshape(reg_val, (-1, nd, n_bins**nd))
    
    best_class = tf.argmin(tf.norm(tfcenters[None] - tf_y[:,:,None], axis=1), axis=-1)
    cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=best_class,
            logits=cls_logit)
    bucket = tf.logical_and(
            tfsections[None,0] < tf_y[:,:,None],
            tfsections[None,1] > tf_y[:,:,None],)
    reg_loss = tf.multiply(
            tf.squared_difference(reg_val + tfcenters[None], tf_y[:,:,None]),
            tf.cast(bucket, tf.float32))
    loss = tf.reduce_mean(cls_loss) + tf.reduce_mean(reg_loss)

    opt = tf.train.AdamOptimizer(1e-3)
    train_op = opt.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(train_steps):
            batch_x, batch_y, _ = make_batch(nd)
            _, l = sess.run([train_op, loss],
                      feed_dict={tf_x: batch_x[:,None],
                                  tf_y: batch_y})
            if i%100==0:
                print(i, l)
        pred_y = [[], []]
        for i in range(100):
            batch_x, batch_y, batch_z = make_batch(nd, batch_size=10)
            c, r = sess.run([cls_probs, reg_val],
                          feed_dict={tf_x: batch_x[:,None],
                                    tf_y: batch_y})
            pr = [joint_mb_sampler(_c, _r, centers) 
                  for _c, _r in zip(c, r)]
            for p, z in zip(pr, batch_z):
                pred_y[z] += [p]
    pred_y = [np.stack(i, -1) for i in pred_y]
    return pred_y

if __name__ == '__main__':
    nd = 4
    np.random.seed(0)
    tf.random.set_random_seed(0)
    l2_pred_y = run_l2_experiment(nd)
    mmr_pred_y = run_mmr_experiment(nd)
    mb_pred_y_ind = run_mb_ind_experiment(nd)
    mb_pred_y_joint = run_mb_joint_experiment(nd)
    
    x, y, z = make_batch(nd, batch_size=1000)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot3d(ax, y[z.astype(bool), 0], y[z.astype(bool), 1], vis_bins=30)
    plot3d(ax, y[~z.astype(bool), 0], y[~z.astype(bool), 1], vis_bins=30)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot3d(ax, *l2_pred_y[1][:2], vis_bins=30)
    plot3d(ax, *l2_pred_y[0][:2], vis_bins=30)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot3d(ax, *mb_pred_y_ind[1][:2], vis_bins=30)
    plot3d(ax, *mb_pred_y_ind[0][:2], vis_bins=30)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot3d(ax, *mb_pred_y_joint[1][:2], vis_bins=30)
    plot3d(ax, *mb_pred_y_joint[0][:2], vis_bins=30)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot3d(ax, *mmr_pred_y[1][:2], vis_bins=30)
    plot3d(ax, *mmr_pred_y[0][:2], vis_bins=30)