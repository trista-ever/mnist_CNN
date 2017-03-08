import input_data
import tensorflow as tf 

#import tensorflow.examples.tutorials.mnist.input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


x=tf.placeholder(tf.float32,[None,784])

w= tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,w)+b)

#loss function
y_=tf.placeholder("float",[None,10])
cross_entropy= - tf.reduce_sum(y_*tf.log(y))

#随机梯度下降
train_step= tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化所有变量
init =tf.initialize_all_variables()

#sess=tf.Session  session 换成interactiveSession就可以了？？？

sess = tf.InteractiveSession()
sess.run(init)

#训练模型
for i in range(1000):
    banch_xs,batch_ys =mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:banch_xs,y_:batch_ys})

#评估模型
correct_prediction= tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

accuracy= tf.reduce_mean(tf.cast(correct_prediction,"float"))

print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))

