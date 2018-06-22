import os
import random
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt


# Normalize by column (min-max norm)
def normalize_cols(m):

    col_max = np.array([110.0 , 200.0 ,130.0])
    col_min = np.array([40.0 , 140.0 , 60.0])
    # print ('col_max = ', col_max )
    # print ('col_min = ', col_min )
    return (m-col_min) / (col_max - col_min)


def predict_weight(testDatas):

    batch_size = 1
    testDatas = np.array(testDatas)

    x_vals_test = normalize_cols(testDatas)
    print('x_vals_test', x_vals_test)
    # Create graph session

    tf.reset_default_graph()
    sess = tf.Session()

    # Define Variable Functions (weights and bias)
    def init_weight(shape, st_dev):
        weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
        return weight

    def init_bias(shape, st_dev):
        bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
        return bias

    # Create a fully connected layer:
    def fully_connected(input_layer, weights, biases):
        layer = tf.add(tf.matmul(input_layer, weights), biases)
        return tf.nn.relu(layer)

    # Initialize placeholders
    x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 2], dtype=tf.float32)

    weight_1 = init_weight(shape=[3, 10], st_dev=1.0)
    bias_1 = init_bias(shape=[10], st_dev=1.0)
    layer_1 = fully_connected(x_data, weight_1, bias_1)

    weight_2 = init_weight(shape=[10, 15], st_dev=1.0)
    bias_2 = init_bias(shape=[15], st_dev=1.0)
    layer_2 = fully_connected(layer_1, weight_2, bias_2)

    weight_3 = init_weight(shape=[15, 15], st_dev=1.0)
    bias_3 = init_bias(shape=[15], st_dev=1.0)
    layer_3 = fully_connected(layer_2, weight_3, bias_3)

    weight_4 = init_weight(shape=[15, 15], st_dev=1.0)
    bias_4 = init_bias(shape=[15], st_dev=1.0)
    layer_4 = fully_connected(layer_3, weight_4, bias_4)

    weight_5 = init_weight(shape=[15, 8], st_dev=1.0)
    bias_5 = init_bias(shape=[8], st_dev=1.0)
    layer_5 = fully_connected(layer_4, weight_5, bias_5)

    weight_6 = init_weight(shape=[8, 2], st_dev=1.0)
    bias_6 = init_bias(shape=[2], st_dev=1.0)

    final_output = tf.sigmoid(tf.add(tf.matmul(layer_5, weight_6), bias_6))

    # batch_num = tf.shape(final_output)[0]
    batch_num = batch_size
    predict_weight = tf.strided_slice(final_output, [0, 0], [batch_num, 1], [1, 1])
    y_target_weght = tf.strided_slice(y_target, [0, 0], [batch_num, 1], [1, 1])

    predict_waist = tf.strided_slice(final_output, [0, 1], [batch_num, 2], [1, 1])
    y_target_waist = tf.strided_slice(y_target, [0, 1], [batch_num, 2], [1, 1])

    weight_loss = tf.reduce_mean(tf.square(y_target_weght - predict_weight))
    waist_loss = tf.reduce_mean(tf.square(y_target_waist - predict_waist))
    # Declare loss function (MSE)
    loss = tf.add(waist_loss, weight_loss)

    # This is caculate the accuracy
    Temp1 = tf.abs(tf.subtract(predict_weight, y_target_weght))
    accuracy_weight = tf.reduce_mean(tf.cast(tf.less(Temp1, 0.12), tf.float32))
    #
    # #This is for caculate the new accurcy
    # Temp2 = tf.subtract(y_target , final_output)

    Temp2 = tf.abs(tf.subtract(predict_waist, y_target_waist))
    accuracy_waist = tf.reduce_mean(tf.cast(tf.less(Temp2, 0.3), tf.float32))

    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)
    # Training loop
    checkpoint_dir = '/home/shen/Trying/Predict/up/Two_out_Net/Net_save3'
    saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        pass
    x = x_vals_test
    # y = np.transpose([testLabels])
    predict_weight_lost = sess.run(predict_weight, feed_dict={x_data: x})
    predict_weight_lost *= 10

    predict_waist_lost = sess.run(predict_waist, feed_dict={x_data: x})
    predict_waist_lost *= 10

    #predict_weight_lost = predict_weight_lost[0][0]
    #print (predict_weight_lost)

    origin_weight = [testDatas[:, 0]]
    transpose_o_weight = np.transpose(origin_weight)
    predict_weight_old = transpose_o_weight - predict_weight_lost
    output_weight = predict_weight_old[0][0]


    origin_waist = [testDatas[:, 2]]
    transpose_o_waist = np.transpose(origin_waist)
    predict_waist_old = transpose_o_waist - predict_waist_lost
    output_waist = predict_waist_old[0][0]
    return (output_weight,output_waist)


output_weight,output_waist = predict_weight([[69.6, 170.0, 80.0]])
print (output_weight )
print (output_waist)


#
# for i in range(3):
#     output_weight = predict_weight([[69.6, 170.0, 80.0]])
#     print (output_weight)
#     time.sleep(2)