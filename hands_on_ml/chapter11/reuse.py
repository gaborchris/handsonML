import __init__
import tensorflow as tf
import src.load_data.loader as data_loader
from functools import partial

def retrain_model(X_train, y_train, X_val, y_val):
    learning_rate = 0.01
    n_inputs = 28*28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name='y')

    training = tf.placeholder_with_default(False, shape=(), name="training")
    my_batch_norm_layer = partial(tf.layers.batch_normalization, training=training, momentum=0.9)

    with tf.name_scope("dnn"):
        he_init = tf.contrib.layers.variance_scaling_initializer()
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
        bn1 = my_batch_norm_layer(hidden1)
        bn1_act = tf.nn.elu(bn1)
        hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2", kernel_initializer=he_init)
        bn2 = my_batch_norm_layer(hidden2)
        bn2_act = tf.nn.elu(bn2)
        logits_before = tf.layers.dense(bn2_act, n_outputs, name="outputs")
        logits = my_batch_norm_layer(logits_before)

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        training_op = optimizer.apply_gradients(grads_and_vars)
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)#, scope='hidden[12]')
    reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
    restore_saver = tf.train.Saver(reuse_vars_dict)

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    init = tf.global_variables_initializer()

    n_epochs = 20
    batch_size = 50

    with tf.Session() as sess:
        sess.run(init)
        restore_saver.restore(sess, tf.train.latest_checkpoint("./"))
        for epoch in range(n_epochs):
            for batch_index in range(int(y_train.shape[0]/batch_size)):
                b_start, b_end = batch_size*batch_index, batch_size*batch_index + batch_size
                X_batch, y_batch = X_train[b_start: b_end], y_train[b_start: b_end]
                sess.run([training_op, extra_update_ops], feed_dict={training: True, X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_val = accuracy.eval(feed_dict={X: X_val, y: y_val})
            print("Epoch: ", epoch, "training: ", acc_train, "val_acc", acc_val)


def load_model_and_run(X_train, y_train, X_val, y_val):
    saver = tf.train.import_meta_graph("./my_model.ckpt.meta")
    #for op in tf.get_default_graph().get_operations():
    #    print(op.name)
    '''
    X = tf.get_default_graph().get_tensor_by_name("X:0")
    y = tf.get_default_graph().get_tensor_by_name("y:0")
    accuracy = tf.get_default_graph().get_tensor_by_name("eval/accuracy:0")
    training_op = tf.get_default_graph().get_operation_by_name("train/GradientDescent")
    '''
    X, y, accuracy, training_op = tf.get_collection("my_important_ops")
    training = tf.placeholder_with_default(False, shape=(), name="training")
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    n_epochs = 15
    batch_size = 100

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint("./"))
        for epoch in range(n_epochs):
            for batch_index in range(int(y_train.shape[0]/batch_size)):
                b_start, b_end = batch_size*batch_index, batch_size*batch_index + batch_size
                X_batch, y_batch = X_train[b_start: b_end], y_train[b_start: b_end]
                sess.run([training_op, extra_update_ops], feed_dict={training: True, X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_val = accuracy.eval(feed_dict={X: X_val, y: y_val})
            print("Epoch: ", epoch, "training: ", acc_train, "val_acc", acc_val)



if __name__ =="__main__":
    image_path = data_loader.dataset_path("mnist", "mnist.npz")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=image_path)
    n_features = 28*28
    x_train = x_train.reshape(-1, n_features)
    x_train = x_train/255
    x_test = x_test.reshape(-1, n_features)
    x_test = x_test/255

    #load_model_and_run(x_train, y_train, x_test,y_test)
    retrain_model(x_train, y_train, x_test,y_test)
