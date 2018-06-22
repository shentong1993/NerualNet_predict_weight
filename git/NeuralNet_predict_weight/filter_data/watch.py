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

trainDataList = processData(filePath='./DataBase/trainGao.csv')
#print ("len = " , len(trainDataList))
#height0,weight0,weight1,weight2,weight3,weight4,waistline0,waistline1,waistline2,waistline3,waistline4,deltaWeight1,deltaWeight2,deltaWeight3,deltaWeight4,deltaWeightAll,deltaWaist1,deltaWaist2,deltaWaist3,deltaWaist4,deltaWaistAll,hipline0,hipline1,hipline2,hipline3,hipline4,chest0,chest1,chest2,chest3,chest4,thigh0,thigh1,thigh2,thigh3,thigh4,arm0,arm1,arm2,arm3,arm4

bmiList = []
hiplineList = []

weightList = []
heightList = []
waistList = []
deltaWeightList  =[]
deltaWaistList = []
for person in trainDataList:
    weight = person['weight0']
    height = person['height0']
    bmi = round(weight / ((height / 100) * (height / 100)), 2)
    waist = person['waistline0']
    deltaWeight = person['deltaWeightAll']
    deltaWaist = person['deltaWaistAll']

    weightList.append(weight)
    heightList.append(height)
    waistList.append(waist)
    deltaWeightList.append(deltaWeight)
    deltaWaistList.append(deltaWaist)

    # if arm > 43 or arm < 18:
    #     trainDataList.remove(person)
    #     continue
    # if bmi >19 and bmi <22 and arm > 33:
    #     trainDataList.remove(person)
    #     continue

    # bmiList.append(bmi)
    # hiplineList.append(arm)


weightList = np.array(weightList)
print ("weight : max = ",np.max(weightList)," "," min = ",np.min(weightList))

heightList = np.array(heightList)
print ("height : max = ",np.max(heightList)," "," min = ",np.min(heightList))

waistList = np.array(waistList)
print ("waist : max = ",np.max(waistList)," "," min = ",np.min(waistList))

deltaWeightList = np.array(deltaWeightList)
print ("delta weight  : max = ",np.max(deltaWeightList)," "," min = ",np.min(deltaWeightList))

deltaWaistList = np.array(deltaWaistList)
print ("waist : max = ",np.max(deltaWaistList)," "," min = ",np.min(deltaWaistList))

print ("len = " , len(trainDataList))
#
# title('X:bmi    Y:arm')
# plot(bmiList, hiplineList, 'r*')

show()


# with open("./data/f7.csv", "w", encoding='utf8') as csvfile:
#     writer = csv.writer(csvfile)
#
#         # 先写入columns_name
#     writer.writerow(['checkinCount1','checkinCount2','checkinCount3','checkinCount4','height0','weight0','weight1','weight2','weight3','weight4','waistline0','waistline1','waistline2','waistline3','waistline4','deltaWeight1','deltaWeight2','deltaWeight3','deltaWeight4','deltaWeightAll','deltaWaist1','deltaWaist2','deltaWaist3','deltaWaist4','deltaWaistAll','hipline0','hipline1','hipline2','hipline3','hipline4','chest0','chest1','chest2','chest3','chest4','thigh0','thigh1','thigh2','thigh3','thigh4','arm0','arm1','arm2','arm3','arm4'])
#
#     for person in trainDataList:
#         writer.writerow([person['checkinCount1'],person['checkinCount2'],person['checkinCount3'],person['checkinCount4'],person['height0'],person['weight0'],person['weight1'],person['weight2'],person['weight3'],person['weight4'],person['waistline0'],person['waistline1'],person['waistline2'],person['waistline3'],person['waistline4'],person['deltaWeight1'],person['deltaWeight2'],person['deltaWeight3'],person['deltaWeight4'],person['deltaWeightAll'],person['deltaWaist1'],person['deltaWaist2'],person['deltaWaist3'],person['deltaWaist4'],person['deltaWaistAll'],person['hipline0'],person['hipline1'],person['hipline2'],person['hipline3'],person['hipline4'],person['chest0'],person['chest1'],person['chest2'],person['chest3'],person['chest4'],person['thigh0'],person['thigh1'],person['thigh2'],person['thigh3'],person['thigh4'],person['arm0'],person['arm1'],person['arm2'],person['arm3'],person['arm4']])
