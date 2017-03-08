import tensorflow as tf 

state = tf.Variable(0, name="Counter")

one =tf.constant(1)
new_value = tf.add(state,one)
update=tf.assign(state,new_value)

init_op=tf.initialize_all_variables()

#tf.global_variables_initializer

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))

    for _ in range(3):
       sess.run(update)
       result=sess.run([state,new_value])  #sess.run(new_value)多计算了一遍
       print(result)
