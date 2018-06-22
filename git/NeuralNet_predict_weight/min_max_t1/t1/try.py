from sklearn import preprocessing
import numpy as np

# 创建一组特征数据，每一行表示一个样本，每一列表示一个特征
x = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -1.]])

xNew = [[0. , 1., -1.]]

min_max_scaler = preprocessing.MinMaxScaler()
x_minmax = min_max_scaler.fit_transform(x)
print (x_minmax)

x_test = np.array([[1.1, -1., 2.]])
x_test_minmax = min_max_scaler.transform(x_test)
print (x_test_minmax)
print (type(x_test_minmax))