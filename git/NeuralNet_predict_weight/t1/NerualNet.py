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




testDataList = processData(filePath='/home/shen/Trying/Predict/up/t1/data2/test.csv')

(testDatas, testLabels) = generateDataAndLabel(type='weight', metaDatas=testDataList, week=1)
testDatas = np.array(testDatas)
testLabels = np.array(testLabels)
testLabels /= 10

x_vals_test = np.nan_to_num(normalize_cols(testDatas))

# Create graph session
sess = tf.Session()



# Declare batch size
batch_size = 3000
#batch_size = 2200

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
Temp =tf.abs( tf.subtract(final_output ,y_target))
accuracy =  tf.reduce_mean( tf.cast( tf.less(Temp ,0.12) ,tf.float32))


# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.005)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
checkpoint_dir = '/home/shen/Trying/Predict/up/t1/Net_save2/'
saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b

loss_vec = []
test_loss = []
for i in range(60000):
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
saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i+1)

x = x_vals_test
y = np.transpose([testLabels])
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x_data: x,y_target: y,}
                                      ))

# Plot loss (MSE) over time
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
