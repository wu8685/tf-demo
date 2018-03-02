import tensorflow as tf

if __name__ == "__main__":
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0)
    print(node1, node2)

    sess = tf.Session()
    print(sess.run([node1, node2]))

    node3 = tf.add(node1, node2)
    print(node3)
    print(sess.run(node3))

    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b

    print(sess.run(adder_node, {a: 3, b: 4.5}))
    print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

    W = tf.Variable([-1.])
    b = tf.Variable(2.)
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    model = W*x + b
    loss = tf.reduce_sum(tf.square(y - model))

    init = tf.global_variables_initializer()
    sess.run(init)

    print("result: ", sess.run(model, {x: [1, 2, 3, 4]}))

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    for i in range(1000):
        sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2 ,-3]})

    cur_W, cur_b, cur_loss = sess.run([W, b, loss], {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
    print("result: ", sess.run([W, b]))
    print("result: W: %s, b: %s, loss: %s"%(cur_W, cur_b, cur_loss))