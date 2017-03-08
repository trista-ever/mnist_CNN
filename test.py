import tensorflow as tf


sess=tf.InteractiveSession()

x=tf.Variable([1.0,2.0])
a=tf.constant([3.0,3.0])

x.initializer.run()

#tf sub

sub=tf.subtract(x,a)

#python3 sub.eval()
print(eval(sub))