import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from pylab import *
import csv

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
    deltaWeightAll = []
    bmiLables  = []
    w0 = []
    w4 = []

    useableKeys = FEATURESMAP[week]

    for metaData in metaDatas:
        trainData = []
        for key in useableKeys:
            trainData.append(metaData[key])
        deltaWeight = metaData[LABELMAP[week]]

        bmi = round(metaData['weight0'] / ((metaData['height0'] / 100) * (metaData['height0'] / 100)), 2)

        trainDatas.append(trainData)
        trainLabels.append(deltaWeight)
        deltaWeightAll.append(metaData['deltaWeightAll'])
        bmiLables.append(bmi)
        w0.append(metaData['weight0'])
        w4.append(metaData['weight4'])

    return (trainDatas, trainLabels, deltaWeightAll,bmiLables,w0 ,w4)


# Normalize by column (min-max norm)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)



#[{ all values of a person },{}...]
# trainDataList = processData(filePath='/home/shen/Trying/Predict/up/t1/data/train.csv')
#
# (trainDatas, trainLabels,deltaWeightAll, bmiLables,weightLables) = generateDataAndLabel(type='weight', metaDatas=trainDataList, week=1)
#
# trainDatas = np.array(trainDatas)
# trainLabels = np.array(trainLabels)
# trainLabels /= 10
#
# x_vals_train = np.nan_to_num(normalize_cols(trainDatas))



testDataList = processData(filePath='/home/shen/Trying/Predict/up/t1/data/trainData.csv')

(testDatas, testLabels ,deltaWeightAll, bmiLables,w0 ,w4) = generateDataAndLabel(type='weight', metaDatas=testDataList, week=1)
testDatas = np.array(testDatas)
testLabels = np.array(testLabels)
testLabels /= 10

x_vals_test = np.nan_to_num(normalize_cols(testDatas))
#x_vals_test = normalize_cols(testDatas)

# Create graph session
sess = tf.Session()



# Initialize placeholders
x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables for both NN layers
hidden_layer_nodes = 10
A1 = tf.Variable(tf.random_normal(shape=[3, hidden_layer_nodes]))  # inputs -> hidden nodes
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))  # one biases for each hidden node

A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, hidden_layer_nodes]))
b2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))  # one biases for each hidden node

A3 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, hidden_layer_nodes]))
b3 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))  # one biases for each hidden node

A4 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 8]))
b4 = tf.Variable(tf.random_normal(shape=[8]))  # one biases for each hidden node

A5 = tf.Variable(tf.random_normal(shape=[8, 1]))  # hidden inputs -> 1 output
b5 = tf.Variable(tf.random_normal(shape=[1]))   # 1 bias for the output

# Declare model operations
hidden_output1 = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
hidden_output2 = tf.nn.relu(tf.add(tf.matmul(hidden_output1, A2), b2))
hidden_output3 = tf.nn.relu(tf.add(tf.matmul(hidden_output2, A3), b3))
hidden_output4 = tf.nn.relu(tf.add(tf.matmul(hidden_output3, A4), b4))

final_output = tf.sigmoid(tf.add(tf.matmul(hidden_output4, A5), b5))

# Declare loss function (MSE)
loss = tf.reduce_mean(tf.square(y_target - final_output))


# This is caculate the accuracy
Temp_real = tf.subtract(final_output ,y_target)

Temp =tf.abs( tf.subtract(final_output ,y_target))
accuracy =  tf.reduce_mean( tf.cast( tf.less(Temp ,0.12) ,tf.float32))


# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.005)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
checkpoint_dir = '/home/shen/Trying/Predict/up/t1/Net_save3_more_accurcy/'
saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    pass

x = x_vals_test
y = np.transpose([testLabels])

final_output = sess.run(final_output, feed_dict={x_data: x, y_target: y})

final_output = final_output.transpose()
final_output = final_output.squeeze()
final_output *= 10

w0 = np.array(w0)
predictWeight = w0 - final_output

print (predictWeight)
# predictWeight = round(predictWeight,2)

w4 = np.array(w4)

print (w0.shape)
print (w4.shape)
print (predictWeight.shape)





Temp_real = sess.run(Temp_real, feed_dict={x_data: x, y_target: y})
# #print( Temp_real )
# print("Testing Accuracy:", sess.run(accuracy, feed_dict={x_data: x, y_target: y, }))
#print (type(Temp_real))
#
# print (Temp_real.shape)
#
Temp_real = Temp_real.transpose()
#
#This is [ , , ,]
Temp_real = Temp_real.squeeze()
# Label_01 = Temp
#
# Draw_y = np.array(deltaWeightAll)

label1_2_3 =[]
for i in range(w0.shape[0]):
    if Temp_real[i] < -0.12:
        label1_2_3.append(1)
    elif abs(Temp_real[i]) < 0.12:
        label1_2_3.append(2)
    else:
        label1_2_3.append(3)
    predictWeight[i] = round(predictWeight[i], 2)

label1_2_3 = np.array(label1_2_3)


#
# #Draw_x = np.arange(Temp.shape[0])
# weightLables = np.array(w0)
# new_y = np.array(deltaWeightAll)
#
# Draw_x = weightLables
#
# # Draw_y = np.array(deltaWeightAll)
# # Draw_x = np.array(bmiLables)
#
# sum =0
# sum_new =0
#
# save_or_not = []
# save_or_not_i =0


# for i in range(len(Draw_x)):
#     if abs(Temp[i]) < 0.12:
#         plot(Draw_x[i], Draw_y[i], 'r*')
#         sum +=1
#
#         save_or_not.append(1)
#     else :
#         plot(Draw_x[i], Draw_y[i], 'b*')
#
#         save_or_not.append(0)
#
#     if Label_01[i] < 0.1:
#         sum_new +=1
#
# print ("save_or_not = ",save_or_not)



with open("/home/shen/Trying/Predict/up/t1/Send_for_other/send.csv", "w", encoding='utf8') as csvfile:
    writer = csv.writer(csvfile)

        # 先写入columns_name
    writer.writerow([
            'weight0', 'weight4','predictWeight' , 'label1_2_3'])

    for i in range(w0.shape[0]):
        writer.writerow([w0[i], ' ',w4[i],' ',predictWeight[i], ' ',label1_2_3[i] ])






# sum = 0
# for i in range(len(Draw_x)):
#     if Draw_y[i] < 0.12:
#         plot(Draw_x[i], new_y[i], 'r*')
#         sum +=1
#     else :
#         plot(Draw_x[i], new_y[i], 'b*')
#
# print ("New_accuracy = ",sum /len(Draw_x) *100)
# print ("New_accuracy1 = ",sum_new /len(Draw_x) *100)
#
#show()




