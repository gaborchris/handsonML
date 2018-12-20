import tensorflow as tf
from functools import partial

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

learning_rate = 0.01

if __name__ == "__main__":
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=None, name="y")
    training = tf.placeholder_with_default(False, shape=(), name='training')
    my_batch_norm_layer = partial(tf.layers.batch_normalization, training=training, momentum=0.9)

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
        bn1 = my_batch_norm_layer(hidden1)
        bn1_act = tf.nn.elu(bn1)
        hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden")
        bn2 = my_batch_norm_layer(hidden1)
        bn2_act = tf.nn.elu(bn2)
        logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs")
        logits = my_batch_norm_layer(logits_before_bn)

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    saver = tf.train.Saver()