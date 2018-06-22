# import csv
# import numpy as np
#
# FEATURESMAP = {
#     1: ['weight0', 'height0', 'waistline0'],
#     2: ['weight1', 'height0', 'waistline1'],
#     3: ['weight2', 'height0', 'waistline2'],
#     4: ['weight3', 'height0', 'waistline3']
# }
#
# def processData(filePath):
#
#     dataList = []
#
#     with open(filePath, 'r') as f:
#         records = f.readlines()
#
#         # 读第一行获取每一列的列名
#         # strip() 去掉行尾的换行符
#         keys = records[0].strip().split(',')
#
#         for i, record in enumerate(records):
#             if i > 0:
#                 dic = {}
#                 values = record.strip().split(',')
#                 for index, key in enumerate(keys):
#                     dic[key] = float(values[index])
#
#                 dataList.append(dic)
#
#     return dataList
#
# def generateDataAndLabel(type, metaDatas, week):
#     LABELMAP = {}
#     if type == 'weight':
#         LABELMAP = {
#             1: 'deltaWeightAll',
#             2: 'deltaWeight2',
#             3: 'deltaWeight3',
#             4: 'deltaWeight4'
#         }
#     elif type == 'waist':
#         LABELMAP = {
#             1: 'deltaWaistAll',
#             2: 'deltaWaist2',
#             3: 'deltaWaist3',
#             4: 'deltaWaist4'
#         }
#     trainDatas = []
#     trainLabels = []
#
#     useableKeys = FEATURESMAP[week]
#
#     for metaData in metaDatas:
#         trainData = []
#         for key in useableKeys:
#             trainData.append(metaData[key])
#         deltaWeight = metaData[LABELMAP[week]]
#
#
#         trainDatas.append(trainData)
#         trainLabels.append(deltaWeight)
#
#     return (trainDatas, trainLabels)
#
# # Normalize by column (min-max norm)
# def normalize_cols(m):
#     col_max = m.max(axis=0)
#     col_min = m.min(axis=0)
#
#     print (col_max)
#     print (col_min)
#
#     with open("/home/shen/Trying/Predict/up/t1/more_dimension/try/min_max_column.csv", "w", encoding='utf8') as csvfile:
#         writer = csv.writer(csvfile)
#
#         # 先写入columns_name
#         writer.writerow(['weight0', 'height0', 'waistline0'])
#
#         writer.writerow([col_max[0], col_max[1],col_max[2]])
#         writer.writerow([col_min[0], col_min[1], col_min[2]])
#
#     return (m-col_min) / (col_max - col_min)
#
#
# #[{ all values of a person },{}...]
# trainDataList = processData(filePath='/home/shen/Trying/Predict/up/t1/more_dimension/more_dimension_data/trainData.csv')
#
# (trainDatas, trainLabels) = generateDataAndLabel(type='weight', metaDatas=trainDataList, week=1)
#
# trainDatas = np.array(trainDatas)
# trainLabels = np.array(trainLabels)
# trainLabels /= 10
#
# x_vals_train = np.nan_to_num(normalize_cols(trainDatas))

# L = [{'a':1 }, {'a':2}]
#
# for l in L:
#     if l['a'] > 1:
#         L.remove(l)
#
# print (L)

from PIL import Image
from pylab import *



# 随意给的一些点
x = [100, 100, 400, 400]
y = [200, 500, 200, 500]

# 使用红色-星状标记需要绘制的点
plot(x, y, 'r*')

# 将数组中的前两个点进行连线
plot(x[:3], y[:3])

# 添加标题信息
title('Plotting: "empire.jpg"')

# 隐藏坐标轴
# axis('off')

# 显示到屏幕窗口
show()