import os
import random
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pylab import *

#,hipline0,hipline1,hipline2,hipline3,hipline4,chest0,chest1,chest2,chest3,chest4,thigh0,thigh1,thigh2,thigh3,thigh4,arm0,arm1,arm2,arm3,arm4

FEATURESMAP = {
    #1: ['weight0', 'height0', 'waistline0','hipline0','chest0','thigh0','arm0'],
    #1: ['weight0', 'height0'],
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


def normalize_cols(m):
    #[weight0 , height0 , waistline0]
    col_max = np.array([110.0 , 200.0 ,130.0])
    col_min = np.array([40.0 , 140.0 , 60.0])
    return (m-col_min) / (col_max - col_min)


#[{ all values of a person },{}...]
trainDataList = processData(filePath='/home/shen/Trying/Predict/up/Final_version_NerualNet/Weight_net/data/train.csv')

#trainDatas = [[weight0 , height0 , waistline0],[]... ]
#trainLabels =[deltaWeightAll ,.....]
(trainDatas, trainLabels) = generateDataAndLabel(type='weight', metaDatas=trainDataList, week=1)

trainDatas = np.array(trainDatas)
trainLabels = np.array(trainLabels)
trainLabels /= 10

x_vals_train = np.nan_to_num(normalize_cols(trainDatas))




testDataList = processData(filePath='/home/shen/Trying/Predict/up/Final_version_NerualNet/Weight_net/more_dimension_data/f6.csv')

(testDatas, testLabels) = generateDataAndLabel(type='weight', metaDatas=testDataList, week=1)
testDatas = np.array(testDatas)
testLabels = np.array(testLabels)
testLabels /= 10

x_vals_test = np.nan_to_num(normalize_cols(testDatas))



# Create graph session
sess = tf.Session()



# Declare batch size
#batch_size = 3000
#batch_size = 2300
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


# # This is caculate the accuracy
# Temp =tf.abs( tf.subtract(final_output ,y_target))
# accuracy =  tf.reduce_mean( tf.cast( tf.less(Temp ,0.12) ,tf.float32))

Temp =tf.abs( tf.subtract(final_output ,y_target))
Temp = tf.multiply(Temp, 10)
accuracy =  tf.reduce_mean( tf.cast( tf.less(Temp ,1.2) ,tf.float32))

#This is for caculate the new accurcy
Temp2 = tf.subtract(y_target , final_output)

# Declare optimizer
# my_opt = tf.train.GradientDescentOptimizer(0.005)
# train_step = my_opt.minimize(loss)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.005
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,20000, 0.96, staircase=True)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(loss, global_step=global_step)


# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# # Training loop

saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b

loss_vec = []
test_loss = []
for i in range(20000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size, replace=False)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([trainLabels[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(np.sqrt(temp_loss))


    if (i + 1) % 100 == 0:
        print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))


save_dir ='./Net_save/'
saver.save(sess, save_dir + 'model.ckpt')

x = x_vals_test
y = np.transpose([testLabels])
print("Testing Accuracy:", sess.run(accuracy*100, feed_dict={x_data: x,y_target: y}
                                      ))


# F = sess.run(final_output ,feed_dict={x_data: x,y_target: y})
# print (F)
# print (type(F))
#
# print ("max = ",np.max(F)," min = ",np.min(F))


num_all = y.shape[0]
num_right = 0
temp2 = sess.run(Temp2, feed_dict={x_data: x,y_target: y})
print (temp2)
print (type(temp2))




for i in range(num_all):
    if temp2[i][0] >= 0:
        num_right += 1

print ("new accuracy = ", num_right / num_all * 100)



# Plot loss (MSE) over time
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
