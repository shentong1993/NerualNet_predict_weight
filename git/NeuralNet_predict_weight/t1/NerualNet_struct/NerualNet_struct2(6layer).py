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


# Normalize by column (min-max norm)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)


#[{ all values of a person },{}...]
trainDataList = processData(filePath='/home/shen/Trying/Predict/up/t1/data/train.csv')

(trainDatas, trainLabels) = generateDataAndLabel(type='weight', metaDatas=trainDataList, week=1)

trainDatas = np.array(trainDatas)
trainLabels = np.array(trainLabels)
trainLabels /= 10

x_vals_train = np.nan_to_num(normalize_cols(trainDatas))




testDataList = processData(filePath='/home/shen/Trying/Predict/up/t1/data/test.csv')

(testDatas, testLabels) = generateDataAndLabel(type='weight', metaDatas=testDataList, week=1)
testDatas = np.array(testDatas)
testLabels = np.array(testLabels)
testLabels /= 10

x_vals_test = np.nan_to_num(normalize_cols(testDatas))

# Create graph session
sess = tf.Session()



# Declare batch size
#batch_size = 3000
batch_size = 3000

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

weight_4 = init_weight(shape=[10, 10], st_dev=1.0)
bias_4 = init_bias(shape=[10], st_dev=1.0)
layer_4 = fully_connected(layer_3, weight_4, bias_4)

weight_5 = init_weight(shape=[10, 10], st_dev=1.0)
bias_5 = init_bias(shape=[10], st_dev=1.0)
layer_5 = fully_connected(layer_4, weight_5, bias_5)

weight_6 = init_weight(shape=[10, 3], st_dev=1.0)
bias_6 = init_bias(shape=[3], st_dev=1.0)
layer_6 = fully_connected(layer_5, weight_6, bias_6)

weight_7 = init_weight(shape=[3, 1], st_dev=1.0)
bias_7 = init_bias(shape=[1], st_dev=1.0)
#final_output = fully_connected(layer_6, weight_7, bias_7)
final_output = tf.sigmoid(tf.add(tf.matmul(layer_6, weight_7), bias_7))


# Declare loss function (MSE)
loss = tf.reduce_mean(tf.square(y_target - final_output))


# This is caculate the accuracy
Temp =tf.abs( tf.subtract(final_output ,y_target))
accuracy =  tf.reduce_mean( tf.cast( tf.less(Temp ,0.12) ,tf.float32))


# Declare optimizer
# my_opt = tf.train.GradientDescentOptimizer(0.005)
# train_step = my_opt.minimize(loss)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.005
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,5000, 0.96, staircase=True)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(loss, global_step=global_step)


# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# # Training loop
# checkpoint_dir = '/home/shen/Trying/Predict/up/t1/Net_save3_more_accurcy/'
# saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
#
# ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
# if ckpt and ckpt.model_checkpoint_path:
#     saver.restore(sess, ckpt.model_checkpoint_path)
# else:
#     pass



loss_vec = []
test_loss = []
for i in range(21000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size, replace=False)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([trainLabels[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(np.sqrt(temp_loss))



    # test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    # test_loss.append(np.sqrt(test_temp_loss))
    if (i + 1) % 50 == 0:
        print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))
        #print( sess.run(final_output, feed_dict={x_data: rand_x, y_target: rand_y}) )

# save_dir ='/home/shen/Trying/Predict/up/t1/NerualNet_struct/Net_save__get_weight_to_train/'
# saver.save(sess, save_dir + 'model.ckpt', global_step=i+1)


x = x_vals_test
y = np.transpose([testLabels])

print("predict :", sess.run(final_output, feed_dict={x_data: x,y_target: y,}
                                      ))
print("Testing Accuracy:", sess.run(accuracy*100, feed_dict={x_data: x,y_target: y,}
                                      ))



# Plot loss (MSE) over time
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
