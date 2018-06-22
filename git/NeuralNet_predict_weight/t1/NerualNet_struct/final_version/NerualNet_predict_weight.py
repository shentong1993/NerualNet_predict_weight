import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

FEATURESMAP = {
    1: ['weight0', 'height0', 'waistline0'],
    2: ['weight1', 'height0', 'waistline1'],
    3: ['weight2', 'height0', 'waistline2'],
    4: ['weight3', 'height0', 'waistline3']
}

def processData(filePath):

    dataList = []

    with open(filePath, 'r') as f:
        records = f.readlines()

        # 读第一行获取每一列的列名
        # strip() 去掉行尾的换行符
        keys = records[0].strip().split(',')

        for i, record in enumerate(records):
            if i > 0:
                dic = {}
                values = record.strip().split(',')
                for index, key in enumerate(keys):
                    dic[key] = float(values[index])

                dataList.append(dic)

    return dataList

def generateDataAndLabel(type, metaDatas, week):
    LABELMAP = {}
    if type == 'weight':
        LABELMAP = {
            1: 'deltaWeightAll',
            2: 'deltaWeight2',
            3: 'deltaWeight3',
            4: 'deltaWeight4'
        }
    elif type == 'waist':
        LABELMAP = {
            1: 'deltaWaistAll',
            2: 'deltaWaist2',
            3: 'deltaWaist3',
            4: 'deltaWaist4'
        }
    trainDatas = []
    trainLabels = []

    useableKeys = FEATURESMAP[week]

    for metaData in metaDatas:
        trainData = []
        for key in useableKeys:
            trainData.append(metaData[key])
        deltaWeight = metaData[LABELMAP[week]]


        trainDatas.append(trainData)
        trainLabels.append(deltaWeight)

    return (trainDatas, trainLabels)

#
# def normalize_cols(m):
#     max_col = []
#     min_col = []
#     with open('/home/shen/Trying/Predict/up/t1/NerualNet_struct/final_version/min_max_for_predict/min_max_column.csv',
#               'r') as f:
#         records = f.readlines()
#
#         for i, record in enumerate(records):
#             if i == 1:
#                 values = record.strip().split(',')
#                 num = len(values)
#                 for j in range(num):
#                     max_col.append(float(values[j]))
#
#             elif i == 2:
#                 values = record.strip().split(',')
#                 num = len(values)
#                 for j in range(num):
#                     min_col.append(float(values[j]))
#
#     col_max = np.array(max_col)
#     col_min = np.array(min_col)
#
#     print ('m = ', m)
#     print ('col_max = ', col_max )
#     print ('col_min = ', col_min )
#     return (m-col_min) / (col_max - col_min)

# Normalize by column (min-max norm)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)


#[{ all values of a person },{}...]
testDataList = processData(filePath='/home/shen/Trying/Predict/up/t1/NerualNet_struct/final_version/test_data/trainData.csv')

#if predict weight_loss you should feed it [{all values of a person}]
(testDatas, testLabels) = generateDataAndLabel(type='weight', metaDatas=testDataList, week=1)
testDatas = np.array(testDatas)
testLabels = np.array(testLabels)
testLabels /= 10

x_vals_test = np.nan_to_num(normalize_cols(testDatas))

# Create graph session
sess = tf.Session()



# Declare batch size
#batch_size = 3000
batch_size = 2300

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

weight_5 = init_weight(shape=[8,1], st_dev=1.0)
bias_5 = init_bias(shape=[1], st_dev=1.0)

final_output = tf.sigmoid(tf.add(tf.matmul(layer_4, weight_5), bias_5))

# Declare loss function (MSE)
loss = tf.reduce_mean(tf.square(y_target - final_output))


# This is caculate the accuracy
Temp =tf.abs( tf.subtract(final_output ,y_target))
accuracy =  tf.reduce_mean( tf.cast( tf.less(Temp ,0.12) ,tf.float32))

#This is the final weight predict in Neural Net
# Weight0 = tf.strided_slice(x_data,[0,0],[3775,1],[1,1])
# predict_final_output = tf.multiply(final_output , 10.0)
# predict_weight = tf.subtract( Weight0 , predict_final_output)


# Declare optimizer
# my_opt = tf.train.GradientDescentOptimizer(0.005)
# train_step = my_opt.minimize(loss)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.005
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,3000, 0.96, staircase=True)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(loss, global_step=global_step)


# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
checkpoint_dir = '/home/shen/Trying/Predict/up/t1/NerualNet_struct/final_version/Net_save/'
saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    pass

x = x_vals_test
y = np.transpose([testLabels])
print("Testing Accuracy : ", sess.run(accuracy*100, feed_dict={x_data: x,y_target: y,}
                                      ))


# print("predict weight : ", sess.run(predictWeight, feed_dict={x_data: x,y_target: y,}
#                                       ))


predict_weight_lost =sess.run(final_output, feed_dict={x_data: x,y_target: y,}
                                      )

predict_weight_lost *= 10

print (predict_weight_lost)
print (predict_weight_lost.shape)
print (type(predict_weight_lost))

origin_weight = [testDatas[:,0]]

transpose_o_weight = np.transpose(origin_weight)
print (origin_weight)
#print (transpose_o_weight)


predict_weight_old = transpose_o_weight - predict_weight_lost

print (predict_weight_old)

#
# www = sess.run(predict_weight, feed_dict={x_data: x,y_target: y,}
#                                       )
#
# print (www)