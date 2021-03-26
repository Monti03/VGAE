import tensorflow as tf

# loss that considers the error reconstructing some edges
def topological_loss(y_actual, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_actual, y_pred)

# calculate the Kullback-Leibler divergence
def KLD(mu, logvar, n_nodes):
    kl = (0.5 / n_nodes) * \
                  tf.reduce_mean(tf.reduce_sum(1 \
                                               + 2 * logvar \
                                               - tf.square(mu) \
                                               - tf.square(tf.exp(logvar)), 1 ))

    return kl
# get the total loss as topological loss - LK div
def total_loss(y_actual, y_pred, mu, logvar, n_nodes):
    KL = KLD(mu, logvar, n_nodes)
    top_loss = topological_loss(y_actual, y_pred)

    return top_loss - KL 