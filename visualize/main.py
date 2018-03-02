import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]))
    wx_plus_b = tf.matmul(inputs, weights) + biases

    if activation_function is None:
        return wx_plus_b
    else:
        return activation_function(wx_plus_b)

if __name__ == '__main__':
    # make up data
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.25, x_data.shape)
    y_data = np.multiply(np.square(x_data), x_data) * 4 + 1 + noise

    # define the model

    # define input output placeholder
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    # define layers
    layer1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu) # activation function here is to enable the model from linear to nolinear
    # define prediction layer
    prediction_layer = add_layer(layer1, 10, 1)

    # define loss function
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction_layer), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    plt.ion()
    plt.show()
    plt.scatter(x_data, y_data)

    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            prediction_value = sess.run(prediction_layer, feed_dict={xs: x_data})
            curloss = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
            # plot the prediction
            lines = plt.plot(x_data, prediction_value, 'r-', lw=2)
            lines[0].remove()
            print(curloss)
            plt.pause(0.5)