import numpy as np
import tensorflow as tf

# a = tf.constant(20)
# b = tf.constant(10)
#
# result1 = tf.cond(a > b, lambda: a, lambda: b)
# result2 = tf.cond(a < b, lambda: a, lambda: b)
#
# # Initialize all the variables (including parameters) randomly.
# init_op = tf.initialize_all_variables()
#
# sess = tf.InteractiveSession()
# # Run the init_op, evaluate the model outputs and print the results:
# sess.run(init_op)
#
# print(sess.run(a))
# print(sess.run(b))
# print("max value is: %d" % sess.run(result1))
# print("min value is: %d" % sess.run(result2))

# a = tf.constant([11, 12 ,13 , 14],shape=(4,1))
#
# b = tf.constant([10,10 ,10 , 10], shape=(4,1))
#
# d = tf.constant([2,2,2, 2], shape=(4,1))
# c = tf.subtract(a,b)
# result1 = tf.less(c , 2 )
#
#
# init_op = tf.initialize_all_variables()
# sess = tf.InteractiveSession()
# # Run the init_op, evaluate the model outputs and print the results:
# sess.run(init_op)
#
# print(sess.run(a))
# print (sess.run(c))
# print (sess.run(result1))

#
# global_step = tf.Variable(0, trainable=False)
# lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step,
#                                 cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
# momentum = cfg.TRAIN.MOMENTUM
# train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss, global_step=global_step)



# import tensorflow as tf
# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print(sess.run(hello))



# import tensorflow as tf
# import numpy as np
#
# x = tf.placeholder(tf.float32, shape=[None, 1])
# y = 4 * x + 4
#
# w = tf.Variable(tf.random_normal([1], -1, 1))
# b = tf.Variable(tf.zeros([1]))
# y_predict = w * x + b
#
#
# loss = tf.reduce_mean(tf.square(y - y_predict))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
#
# #isTrain = True
# isTrain = False
# train_steps = 100
# checkpoint_steps = 50
# checkpoint_dir = '/home/shen/Trying/Predict/up/t1/Net_save/'
#
# saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
# x_data = np.reshape(np.random.rand(10).astype(np.float32), (10, 1))
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     if isTrain:
#         for i in range(train_steps):
#             sess.run(train, feed_dict={x: x_data})
#             #if (i + 1) % checkpoint_steps == 0:
#         saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i+1)
#     else:
#         ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
#         if ckpt and ckpt.model_checkpoint_path:
#             saver.restore(sess, ckpt.model_checkpoint_path)
#         else:
#             pass
#         print(sess.run(w))
#         print(sess.run(b))
#         y_predict = sess.run(y_predict, feed_dict={x: x_data})
#         print (y_predict)
#         print (type(y_predict))


# list = [{'p1':10}, {'p2':20} , {'p3':30}]
#
# del list[1]
# print (list)

#for l in list :

# n = np.array([[1,2,3],
#               [4,5,6],
#               [7,8,9]])
#
# print (n)
#
# n1 = [n[:,0]]
# print (n1)
# print (type(n1))
#
# n2 = np.transpose(n1)
# print (n2)

# print (n)
# print (n.shape)
#
# col_max = n.max(axis= 0)
#
# print (col_max)
# print (col_max.shape)

# import tensorflow as tf
# arr = tf.Variable(tf.truncated_normal([3,4,1,6,1], stddev=0.1))
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print (sess.run(arr).shape )

# import tensorflow as tf
# import numpy as np
# data = np.array(
#       [[[1, 1, 1], [2, 2, 2]],
#        [[3, 3, 3], [4, 4, 4]],
#        [[5, 5, 5], [6, 6, 6]]]
#         )
# x = tf.strided_slice(data,[0,0,0],[1,1,1])
# with tf.Session() as sess:
#     print (sess.run(x))

#import tensorflow as tf
# t = tf.constant([[[1, 1, 1], [2, 2, 2], [7, 7, 7]],
#                  [[3, 3, 3], [4, 4, 4], [8, 8, 8]],
#                  [[5, 5, 5], [6, 6, 6], [9, 9, 9]]])
#
# z1 = tf.strided_slice(t, [1], [-1], [1])
# z2 = tf.strided_slice(t, [1, 0], [-1, 2], [1, 1])
# z3 = tf.strided_slice(t, [1, 0, 1], [-1, 2, 3], [1, 1, 1])
#
# with tf.Session() as sess:
#     print(sess.run(z1))
#     print (z1.shape)
#     print(sess.run(z2))
#     print(sess.run(z3))


# import tensorflow as tf
#
# t = tf.constant([[1,2,3],
#                  [4,5,6],
#                  [7,8,9]])
# t = tf.cast(t ,tf.float32)
#
# t2 = tf.constant([[0.1],
#                   [0.1],
#                   [0.3]])
#
# t2 = tf.multiply(t2 ,10)
#
# tran_t = tf.strided_slice(t,[0,0],[3,1],[1,1])
#
# t1_sub_t2 = tf.subtract(tran_t ,t2 )
#
# with tf.Session() as sess:
#     print (sess.run(t))
#     print (sess.run(tran_t))
#     print (sess.run(t1_sub_t2))
#     print (sess.run(t2))


import numpy as np

n = np.array([[1,10, 20],
              [3,14, 22],
              [5,17 ,25]])

col_max = n.max(axis=0)

print (col_max)
print (type(col_max))