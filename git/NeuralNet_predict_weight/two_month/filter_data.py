import csv
from PIL import Image
from pylab import *
import numpy as np

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

#trainDataList = processData(filePath='./data_one/trainData.csv')
trainDataList = processData(filePath='./data_two_final/trainDataJJ.csv')
print ("len = " , len(trainDataList))
#height0,weight0,weight8,waistline0,waistline8,deltaWeightAll,deltaWaistAll,hipline0,hipline8,chest0,chest8,thigh0,thigh8,arm0,arm8

bmiList = []

weightList = []
heightList = []
waistlineList = []
deltaWeightAllList = []
deltaWaistAllList = []

for person in trainDataList:
    weight = person['weight0']
    height = person['height0']
    bmi = round(weight / ((height / 100) * (height / 100)), 2)
    waistline = person['waistline0']
    deltaWeightAll = person['deltaWeightAll']
    deltaWaistAll = person['deltaWaistAll']

    # if weight > 110 :
    #     trainDataList.remove(person)
    #     continue

    weightList.append(weight)
    heightList.append(height)
    waistlineList.append(waistline)
    deltaWeightAllList.append(deltaWeightAll)
    deltaWaistAllList.append(deltaWaistAll)

    bmiList.append(bmi)


title('X:bmi    Y:hipline')
plot(bmiList, deltaWaistAllList, 'r*')

weightList = np.array(weightList)
print ("weigth : max = ", np.max(weightList) , " min = ",np.min(weightList))

heightList = np.array(heightList)
print ("height : max = ", np.max(heightList) , " min = ",np.min(heightList))

waistlineList = np.array(waistlineList)
print ("waist : max = ", np.max(waistlineList) , " min = ",np.min(waistlineList))

deltaWeightAllList = np.array(deltaWeightAllList)
print ("deltaWeight : max = ", np.max(deltaWeightAllList) , " min = ",np.min(deltaWeightAllList))

deltaWaistAllList = np.array(deltaWaistAllList)
print ("deltaWaist : max = ", np.max(deltaWaistAllList) , " min = ",np.min(deltaWaistAllList))

print ("len = " , len(trainDataList))



show()

#
# with open("./data_two_final/trainDataJJ.csv", "w", encoding='utf8') as csvfile:
#     writer = csv.writer(csvfile)
#
# # height0,weight0,weight8,waistline0,waistline8,deltaWeightAll,deltaWaistAll,hipline0,hipline8,chest0,chest8,thigh0,thigh8,arm0,arm8
#
#     # 先写入columns_name
#     writer.writerow(['height0',
#                      'weight0',
#                      'weight8',
#                      'waistline0',
#                      'waistline8',
#                      'deltaWeightAll',
#                      'deltaWaistAll',
#                      'hipline0',
#                      'hipline8',
#                      'chest0','chest8','thigh0','thigh8','arm0','arm8'])
#
#     for person in trainDataList:
#         writer.writerow([person['height0'],
#                          person['weight0'],
#                          person['weight8'],
#                          person['waistline0'],
#                          person['waistline8'],
#                          person['deltaWeightAll'],
#                          person['deltaWaistAll'],
#                          person['hipline0'],
#                          person['hipline8'],
#                          person['chest0'],person['chest8'],person['thigh0'],person['thigh8'],person['arm0'],person['arm8']])
