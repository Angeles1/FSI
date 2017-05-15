import gzip
import cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set


# ---------------- Visualizing some element of the MNIST dataset --------------

#import matplotlib.cm as cm
#import matplotlib.pyplot as plt

#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print train_y[57]


# TODO: the neural net!!

#Division del conjunto en subconjuntos : train, validacion, test
train_x, train_y = train_set
train_y = one_hot(train_y.astype(int), 10)

x_validation_data, y_validation_data = valid_set
y_validation_data = one_hot(y_validation_data.astype(int), 10)

x_test_data, y_test_data = test_set
y_test_data = one_hot(y_test_data.astype(int), 10)




x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 100)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(100)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(100, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

#loss = tf.reduce_sum(tf.square(y_ - y))
loss = -tf.reduce_sum(y_ * tf.log(y))


train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

batch_size = 20

error_actual_validacion=1
error_previo_validacion=-1
epoch=0

print "----------------------------------------------------------------------------------"
print "   ENTRENAMIENTO    "
print "----------------------------------------------------------------------------------"

while abs(error_actual_validacion - error_previo_validacion) > 1:
#for epoch in xrange(100):
    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    epoch+=1
    error_previo_validacion=error_actual_validacion
    error_actual_validacion = sess.run(loss, feed_dict={x: x_validation_data, y_: y_validation_data})

    print "----------------------------------------------------------------------------------"
    print "   VALIDACION    "
    print "----------------------------------------------------------------------------------"
    print "Epoch #:", epoch, "Error: ", error_actual_validacion
    #result = sess.run(y, feed_dict={x: x_validation_data})
#    for b, r in zip(batch_ys, result):
#        print b, "-->", r
print "----------------------------------------------------------------------------------"



print "----------------------------------------------------------------------------------"
print "   TEST    "
print "----------------------------------------------------------------------------------"
result = sess.run(y, feed_dict={x: x_test_data})
#error  = sess.run(loss, feed_dict={x: x_test_data, y_: y_test_data})
num_aciertos = 0
num_fallos = 0
for b, r in zip(y_test_data, result):
    if np.array_equal(b,  .round(0)):
        num_aciertos += 1
    else:
        num_fallos += 1
total = num_aciertos+num_fallos
print "Numero de Pruebas = ", total
print "Numero de Aciertos = ", num_aciertos
print "Numero de Fallos = ", num_fallos
print "Error = ", float (num_fallos)/total *100, "%"

print "----------------------------------------------------------------------------------"

