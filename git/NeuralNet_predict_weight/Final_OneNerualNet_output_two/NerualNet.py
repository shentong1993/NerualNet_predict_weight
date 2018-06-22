import os
import random
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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

def generateDataAndLabel(metaDatas):

    trainDatas = []
    trainLabels = []

    useableKeys = ['weight0', 'height0', 'waistline0']
    useableLabels = ['deltaWeightAll', 'deltaWaistAll']

    for metaData in metaDatas:
        trainData = []
        trainLabel = []
        for key in useableKeys:
            trainData.append(metaData[key])


        for key in useableLabels:
            trainLabel.append(metaData[key])

        trainDatas.append(trainData)
        trainLabels.append(trainLabel)

    return (trainDatas, trainLabels)


#Normalize by column (min-max norm)
def normalize_cols(m):
    #[weight0 , height0 , waistline0]
    col_max = np.array([110.0 , 200.0 ,130.0])
    col_min = np.array([40.0 , 140.0 , 60.0])

    return (m-col_min) / (col_max - col_min)

#[{ all values of a person },{}...]
#trainDataList = processData(filePath='/home/shen/Trying/Predict/up/Final_OneNerualNet_output_two/train_data/trainData.csv')
trainDataList = processData(filePath='/home/shen/Trying/Predict/up/Final_OneNerualNet_output_two/test_data/trainData.csv')

(trainDatas, trainLabels) = generateDataAndLabel( metaDatas=trainDataList)

trainDatas = np.array(trainDatas)
trainLabels = np.array(trainLabels)
trainLabels /= 10


x_vals_train = np.nan_to_num(normalize_cols(trainDatas))



testDataList = processData(filePath='/home/shen/Trying/Predict/up/Final_OneNerualNet_output_two/test_data/trainData.csv')

(testDatas, testLabels) = generateDataAndLabel( metaDatas=testDataList)
testDatas = np.array(testDatas)
testLabels = np.array(testLabels)
testLabels /= 10

x_vals_test = np.nan_to_num(normalize_cols(testDatas))



# Create graph session
tf.reset_default_graph()
sess = tf.Session()

# Declare batch size
batch_size = 2723

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

weight_6 = init_weight(shape=[8,2], st_dev=1.0)
bias_6 = init_bias(shape=[2], st_dev=1.0)

final_output = tf.sigmoid(tf.add(tf.matmul(layer_5, weight_6), bias_6))

#batch_num = tf.shape(final_output)[0]
#batch_num = batch_size
batch_num = tf.shape(final_output)[0]
predict_weight = tf.strided_slice(final_output,[0,0],[batch_num,1],[1,1])
y_target_weght = tf.strided_slice(y_target,[0,0],[batch_num,1],[1,1])

predict_waist = tf.strided_slice(final_output,[0,1],[batch_num,2],[1,1])
y_target_waist = tf.strided_slice(y_target,[0,1],[batch_num,2],[1,1])


weight_loss = tf.reduce_mean(tf.square(y_target_weght - predict_weight))
waist_loss = tf.reduce_mean(tf.square(y_target_waist - predict_waist))
#Declare loss function (MSE)
loss = tf.add(waist_loss ,weight_loss)


# This is caculate the accuracy
Temp1 =tf.abs( tf.subtract(predict_weight ,y_target_weght))
accuracy_weight =  tf.reduce_mean( tf.cast( tf.less(Temp1 ,0.12) ,tf.float32))

#This is for caculate the new accurcy
Temp3 = tf.subtract(y_target_weght , predict_weight)

Temp2 =tf.abs( tf.subtract(predict_waist ,y_target_waist))
accuracy_waist =  tf.reduce_mean( tf.cast( tf.less(Temp2 ,0.3) ,tf.float32))

Temp4 = tf.subtract(y_target_waist , predict_waist)


global_step = tf.Variable(0, trainable=False)
#starter_learning_rate = 0.005
starter_learning_rate = 0.005
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,30000, 0.96, staircase=True)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(loss, global_step=global_step)


# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# # Training loop

checkpoint_dir = '/home/shen/Trying/Predict/up/Final_OneNerualNet_output_two/Net_save/'
saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    pass

loss_vec = []
test_loss = []
for i in range(20000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size, replace=False)
    rand_x = x_vals_train[rand_index]
    rand_y = trainLabels[rand_index]

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(np.sqrt(temp_loss))

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})


    if (i + 1) % 100 == 0:
        print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))


saver.save(sess, checkpoint_dir + 'model.ckpt')

x = x_vals_test
y = testLabels
print("weight Testing Accuracy:", sess.run(accuracy_weight*100, feed_dict={x_data: x,y_target: y}
                                      ))
print("waist Testing Accuracy:", sess.run(accuracy_waist*100, feed_dict={x_data: x,y_target: y}
                                      ))
#
# num_all = y.shape[0]
# num_right3 = 0
# temp3 = sess.run(Temp3, feed_dict={x_data: x,y_target: y})
#
#
# for i in range(num_all):
#     if temp3[i][0] >= -0.1:
#         num_right3 += 1
#
# print ("new accuracy weight  = ", num_right3 / num_all * 100)
#
#
#
#
# temp4 = sess.run(Temp4, feed_dict={x_data: x,y_target: y})
#
# num_right4 =0
# for i in range(num_all):
#     if temp4[i][0] >= -0.25:
#         num_right4 += 1
# print ("new accuracy weight  = ", num_right4 / num_all * 100)
#
#Plot loss (MSE) over time
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
