import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from numpy import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print('MINST\SimpleNN')
print('Loading data')
data = pd.read_csv('data/train.csv')
X_train, X_valid, y_train, y_valid = train_test_split(data[data.columns[1:]].as_matrix(),
                                                      data[data.columns[0]].as_matrix(),
                                                      test_size=0.1,
                                                      random_state=1)

enc = OneHotEncoder(sparse=False, n_values=10)
y_train_oh = enc.fit_transform(y_train.reshape(-1, 1))
y_valid_oh = enc.transform(y_valid.reshape(-1, 1))

# Create the model
print('Creating a model...')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)#tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


H = 500
x = tf.placeholder(tf.float32, [None, 784])

# Nonlinear
Wij = weight_variable([784, H])
bj = bias_variable([H])
yj = tf.nn.sigmoid(tf.matmul(x, Wij) + bj)

Wjk = weight_variable([H, 10])
bk = bias_variable([10])
y = tf.matmul(yj, Wjk) + bk

# Linear
#W = tf.Variable(tf.zeros([784, 10]))
#b = tf.Variable(tf.zeros([10]))
#y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Training...')

plt.ion()
fig = plt.figure()
train_line, = fig.gca().plot([], [], color='g', label='train data')
valid_line, = fig.gca().plot([], [], color='r', label='validation data')
fig.gca().legend(loc='lower center')
plt.ylabel('Accuracy')
plt.xlabel('Iteration')


def add_point(x, y, line):
    line.set_xdata(np.append(line.get_xdata(), x))
    line.set_ydata(np.append(line.get_ydata(), y))
    plt.gca().relim()
    plt.gca().autoscale_view()
    plt.pause(0.05)


# Train
for i in range(30000):
    sample = random.choice(X_train.shape[0], 50, replace=False)
    sess.run(train_step, feed_dict={x: X_train[sample], y_: y_train_oh[sample]})

    if i % 1000 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y_: y_train_oh})
        valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y_: y_valid_oh})
        add_point(i, train_accuracy, train_line)
        add_point(i, valid_accuracy, valid_line)

# Test trained model
print('Train score:', sess.run(accuracy, feed_dict={x: X_train, y_: y_train_oh}))
print('Validation score:', sess.run(accuracy, feed_dict={x: X_valid, y_: y_valid_oh}))

# Solving for test data
print('Solving...')
X_test = pd.read_csv('data/test.csv')
result = sess.run(tf.argmax(y, 1), feed_dict={x: X_test})
result = pd.DataFrame(list(result), columns=['Label'])
result.index = np.arange(1, len(result) + 1)
result.index.name = 'ImageId'
result.to_csv('result.csv')
print('Finish!!!')

plt.ioff()
plt.show()


