import os
import random
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt


# Normalize by column (min-max norm)
def normalize_cols(m):
    # max_col = []
    # min_col = []
    # with open('/home/shen/Trying/Predict/up/Final_version_loadAndPredict/normalization/min_max_column.csv',
    #           'r') as f:
    #     records = f.readlines()
    #
    #     for i, record in enumerate(records):
    #         if i == 1:
    #             values = record.strip().split(',')
    #             num = len(values)
    #             for j in range(num):
    #                 max_col.append(float(values[j]))
    #
    #         elif i == 2:
    #             values = record.strip().split(',')
    #             num = len(values)
    #             for j in range(num):
    #                 min_col.append(float(values[j]))
    #
    # col_max = np.array(max_col)
    # col_min = np.array(min_col)
    col_max = np.array([110.0 , 200.0 ,130.0])
    col_min = np.array([40.0 , 140.0 , 60.0])
    # print ('col_max = ', col_max )
    # print ('col_min = ', col_min )
    return (m-col_min) / (col_max - col_min)


def predict_weight(testDatas):
    #tf.reset_default_graph()
    # input data must like this [[weight0 ,height0,waistline0]]
    #testDatas = [[69.6, 170.0, 80.0]]
    # testLabels =[2.1]
    testDatas = np.array(testDatas)
    # testLabels = np.array(testLabels)
    # testLabels /= 10
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
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    weight_1 = init_weight(shape=[3, 10], st_dev=1.0)
    bias_1 = init_bias(shape=[10], st_dev=1.0)
    layer_1 = fully_connected(x_data, weight_1, bias_1)
    weight_2 = init_weight(shape=[10, 10], st_dev=1.0)
    bias_2 = init_bias(shape=[10], st_dev=1.0)
    layer_2 = fully_connected(layer_1, weight_2, bias_2)
    weight_3 = init_weight(shape=[10, 10], st_dev=1.0)
    bias_3 = init_bias(shape=[10], st_dev=1.0)
    layer_3 = fully_connected(layer_2, weight_3, bias_3)
    weight_4 = init_weight(shape=[10, 8], st_dev=1.0)
    bias_4 = init_bias(shape=[8], st_dev=1.0)
    layer_4 = fully_connected(layer_3, weight_4, bias_4)
    weight_5 = init_weight(shape=[8, 1], st_dev=1.0)
    bias_5 = init_bias(shape=[1], st_dev=1.0)
    final_output = tf.sigmoid(tf.add(tf.matmul(layer_4, weight_5), bias_5))
    # Declare loss function (MSE)
    loss = tf.reduce_mean(tf.square(y_target - final_output))
    # This is caculate the accuracy
    Temp = tf.abs(tf.subtract(final_output, y_target))
    accuracy = tf.reduce_mean(tf.cast(tf.less(Temp, 0.12), tf.float32))
    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)
    # Training loop
    checkpoint_dir = '/home/shen/Trying/Predict/up/Final_version_loadAndPredict/Weight_net/Net_save/'
    saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        pass
    x = x_vals_test
    # y = np.transpose([testLabels])
    predict_weight_lost = sess.run(final_output, feed_dict={x_data: x})
    predict_weight_lost *= 10
    # print('predict_weight_lost = ', predict_weight_lost)
    origin_weight = [testDatas[:, 0]]
    # print ('origin_weight = ', origin_weight)
    transpose_o_weight = np.transpose(origin_weight)
    # print ('transpose_o_weight = ', transpose_o_weight)
    predict_weight_old = transpose_o_weight - predict_weight_lost
    output_weight = predict_weight_old[0][0]
    # print(output_weight)
    # print(round(output_weight, 2))

    return round(output_weight, 2)

# output_weight = predict_weight([[69.6, 170.0, 80.0]])
# print (output_weight)



for i in range(3):
    output_weight = predict_weight([[69.6, 170.0, 80.0]])
    print (output_weight)
    time.sleep(2)