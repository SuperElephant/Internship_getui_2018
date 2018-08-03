
"""
This is model to classify  user according to the tags

:author: Chao Xiang
:date: 2018

----first trial (fail)----
1. Convolutional layer #1
2. Pooling Layer #1
3. Convolutional layer #2
4. Pooling Layer #2
5. Dense Layer #1
6. Dense Layer #2
While using this model the loss does not converge and prediction is around 50% which
is meaningless for a binary classification problem.
CNN may fit for this kind of problem, consider that convolution and pooling will
confuse tags with different meanings but adjacent in feature vector

----second trial (present)-----
1. Dense Layer #1
2. Dense Layer #2
Works fine, after 25k steps correction of production converge at 96%
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer

tf.logging.set_verbosity(tf.logging.INFO)


def data_preprocess():
    raw_data0 = open("~/Desktop/sample_data/20180525_ml_data_0.txt", "r")
    raw_data1 = open("~/Desktop/sample_data/20180525_ml_data_1.txt", "r")
    lines0 = raw_data0.readlines()
    lines1 = raw_data1.readlines()
    vectorizer = CountVectorizer(token_pattern=u'[A-Za-z0-9\.]+')
    for i in range(len(lines0)):
        lines0[i] = lines0[i].split('|', 2)[2].strip()
    for i in range(len(lines1)):
        lines1[i] = lines0[i].split('|', 2)[2].strip()

    all_lines = lines0 + lines1

    data = vectorizer.fit_transform(all_lines).toarray().astype(np.float32)
    labels = np.append(np.zeros(shape=1000), np.ones(shape=1000)).astype(np.int32)

    # Manually split test data
    train_features = np.vstack((data[0:700], data[1000:1700]))
    test_features = np.vstack((data[700:1000], data[1700:2000]))

    train_labels = np.append(labels[0:700], labels[1000:1700])
    test_labels = np.append(labels[700:1000], labels[1700:2000])

    return train_features, train_labels, test_features, test_labels


def cnn_model_fn(features, labels, mode):

    # # ----------first trial (fail)-------------
    # # [batch_size, length, channel]
    # input_layer = tf.reshape(features["tages"], [-1, 6396, 1])
    #
    # # [batch_size, 6396, 1]
    # # [batch_size, 6396, 8]
    # conv1 = tf.layers.conv1d(
    #     inputs=input_layer,
    #     filters=8,
    #     kernel_size=32,
    #     padding="same",
    #     activation=tf.sigmoid
    # )
    #
    # # [batch_size, 6396, 8]
    # # [batch_size, 799, 8]
    # pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=8, strides=8)
    #
    # # [batch_size, 799, 8]
    # # [batch_size, 799, 16]
    # conv2 = tf.layers.conv1d(
    #     inputs=pool1,
    #     filters=16,
    #     kernel_size=32,
    #     padding="same",
    #     activation=tf.sigmoid
    # )
    # # [batch_size, 799, 16]
    # # [batch_size, 199, 16]
    #
    # pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=4, strides=4)
    #
    # pool2_flat = tf.reshape(pool2, [-1, 199*16])
    #
    # dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.sigmoid)
    #
    # # Add dropout operation; 0.6 probability that element will be kept
    # dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    #
    # # dense2 = tf.layers.dense(inputs=dropout, units=128, activation=tf.sigmoid)
    #
    # logits = tf.layers.dense(inputs=dropout, units=2)
    # # ============================================

    # -----------second trial------------
    input_layer = tf.reshape(features["tages"], [-1, 6396])

    dense = tf.layers.dense(inputs=input_layer, units=1024, activation=tf.sigmoid)

    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense2 = tf.layers.dense(inputs=dropout, units=128, activation=tf.sigmoid)

    logits = tf.layers.dense(inputs=dense2, units=2)
    # =====================================

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unsed_argv):
    train_data, train_labels, test_data, test_labels = data_preprocess()
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/Users/chaoxiang/Desktop/sample_data/model1"
    )

    tensor_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=50)

    # train 200 * 100 steps
    for i in range(200):
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'tages': train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True
        )

        classifier.train(
            input_fn=train_input_fn,
            steps=100,
            hooks=[logging_hook])

        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'tages': test_data},
            y=test_labels,
            num_epochs=1,
            shuffle=False
        )

        test_results = classifier.evaluate(input_fn=test_input_fn)
        print(test_results)


if __name__ == "__main__":
    tf.app.run()
