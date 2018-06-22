from sklearn import preprocessing
import numpy as np

# 创建一组特征数据，每一行表示一个样本，每一列表示一个特征
x = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -1.]])

xNew = [[0. , 1., -1.]]
x_mean = x.mean(axis=0)
x_std = x.std(axis=0)

print("x_mean = ",x_mean )
print ("x_std = ", x_std)
# 将每一列特征标准化为标准正太分布，注意，标准化是针对每一列而言的
x_scale = preprocessing.scale(x)

print (x_scale)


print ("")

scaler=preprocessing.StandardScaler().fit(x)
X=scaler.transform(x)
print (X)
xNew=scaler.transform(xNew)

print (xNew)