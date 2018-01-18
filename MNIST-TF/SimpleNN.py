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
X_test = pd.read_csv('data/test.csv')
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
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


H1 = 784
H2 = 100
C = 0.00001
x = tf.placeholder(tf.float32, [None, 784])

# Nonlinear
Wij = weight_variable([784, H1])
bj = bias_variable([H1])
yj = tf.nn.sigmoid(tf.matmul(x, Wij) + bj)

Wjk = weight_variable([H1, H2])
bk = bias_variable([H2])
yk = tf.nn.sigmoid(tf.matmul(yj, Wjk) + bk)

Wkl = weight_variable([H2, 10])
bl = bias_variable([10])
y = tf.matmul(yk, Wkl) + bl

sizeWij = 784 * H1
sizeWjk = H1 * H2
sizeWkl = H2 * 10
sizeW = sizeWij + sizeWjk + sizeWkl

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
regularization = sizeWij / sizeW * tf.reduce_sum(tf.square(Wij)) + \
                 sizeWjk / sizeW * tf.reduce_sum(tf.square(Wjk)) + \
                 sizeWkl / sizeW * tf.reduce_sum(tf.square(Wkl));

global_step = tf.Variable(0, trainable=False)
boundaries = [5000, 25000, 40000]
values = [5e-4, 1e-4, 5e-5, 1e-5]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy + C * regularization)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Training...')

plt.ion()
fig = plt.figure()
fig.canvas.set_window_title('Train and Validation Accuracy')
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
best_accuracy = 0


def compute_result():
    print('saving results...')
    global y
    global X_test
    result = sess.run(tf.argmax(y, 1), feed_dict={x: X_test})
    result = pd.DataFrame(list(result), columns=['Label'])
    result.index = np.arange(1, len(result) + 1)
    result.index.name = 'ImageId'
    result.to_csv('result.csv')


for i in range(50000):
    sample = random.choice(X_train.shape[0], 50, replace=False)

    sess.run(train_step, feed_dict={x: X_train[sample], y_: y_train_oh[sample]})

    if i % 1000 == 0 and i > 3000:
        train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y_: y_train_oh})
        valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y_: y_valid_oh})

        print('Scores: train', train_accuracy, 'validation', valid_accuracy)

        if valid_accuracy > best_accuracy and i > 25000:
            best_accuracy = valid_accuracy
            compute_result()

        add_point(i, train_accuracy, train_line)
        add_point(i, valid_accuracy, valid_line)

# Test trained model
print('Final train score:', sess.run(accuracy, feed_dict={x: X_train, y_: y_train_oh}))
valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y_: y_valid_oh})
print('Final validation score:', valid_accuracy)

# Solving for test data
if valid_accuracy > best_accuracy:
    compute_result()

print('Finish!!!')

plt.ioff()
plt.show()


