from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
import numpy as np

#print('input data:')
#print(mnist.train.images)
#print('input data shape:')
#print(mnist.train.images.shape)
"""
import pylab 

im = mnist.train.images[1]
im = im.reshape(-1,28)
pylab.imshow(im)
pylab.show()
"""
'''
a=np.asarray(range(20))
b=a.reshape(-1,2,2)

print('生成一列数据')
print(a)
print('reshape函数的效果') 
print(b)

c = np.transpose(b,[1,0,2])
d = c.reshape(-1,2)
print('--------c-----------')
print(c)
print('--------d-----------')
print(d)
'''

# Parameters
learning_rate = 0.001
training_iters = 1000
batch_size = 128
display_step = 100

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

x = tf.placeholder("float32", [None, n_steps, n_input])
y = tf.placeholder("float32", [None, n_classes])

# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=0.0, state_is_tuple=True) 
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=0.0, state_is_tuple=True)
_state = lstm_cell.zero_state(batch_size,tf.float32) #initialization batch_size = 128


a1 = tf.transpose(x, [1, 0, 2])
a2 = tf.reshape(a1, [-1, n_input]) 
a3 = tf.matmul(a2, weights['hidden']) + biases['hidden']   
a4 = tf.split(a3,n_steps, 0)

print('-----------------------') 
print('a1:') 
print(a1) 
print('-----------------------') 

print('a2:') 
print(a2) 
print('-----------------------') 
print('a3:') 
print(a3)
print('-----------------------') 
print('a4:') 
print(a4) 

outputs, states = lstm_cell(a4, _state)
a5 = tf.matmul(outputs[-1], weights['out']) + biases['out']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(a5, y))
#AdamOptimizer
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
correct_pred = tf.equal(tf.argmax(a5,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.initialize_all_variables()

sess = tf.InteractiveSession() 
sess.run(init)
step = 1

while step * batch_size < training_iters:
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
    if step % display_step == 0:
         acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,})
         loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
         print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) +  ", Training Accuracy= " + "{:.5f}".format(acc))
    step += 1

print("Optimization Finished!") 

