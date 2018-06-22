from sklearn import neighbors
import sklearn
import numpy as np
import random


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

        #print (keys)

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
        if week == 1:


            if 0 <= deltaWeight <= 0.5:
                trainLabel = 0
            elif 0.5 <= deltaWeight <= 1:
                trainLabel = 1
            elif 1 <= deltaWeight <= 1.5:
                trainLabel = 2
            elif 1.5 <= deltaWeight <= 2:
                trainLabel = 3
            elif 2 <= deltaWeight <= 2.5:
                trainLabel = 4
            elif 2.5 <= deltaWeight <= 3:
                trainLabel = 5
            elif 3 <= deltaWeight < 3.5:
                trainLabel = 6
            elif 3.5 <= deltaWeight <= 4:
                trainLabel = 7
            elif 4 <= deltaWeight <= 4.5:
                trainLabel = 8
            elif 4.5 <= deltaWeight <= 5:
                trainLabel = 9
            elif 5 <= deltaWeight <= 5.5:
                trainLabel = 10


            elif 5.5 <= deltaWeight <= 6:
                trainLabel = 11
            elif 6 <= deltaWeight <= 6.5:
                trainLabel = 12

            elif 6.5 <= deltaWeight <= 7:
                trainLabel = 13
            elif 7 <= deltaWeight <= 7.5:
                trainLabel = 14
            elif 7.5 <= deltaWeight < 8:
                trainLabel = 15

            elif 8 <= deltaWeight <= 8.5:
                trainLabel = 16
            elif 8.5 <= deltaWeight <= 9:
                trainLabel = 17
            elif 9 <= deltaWeight <= 9.5:
                trainLabel = 18
            else:
                trainLabel =19






        else:
            if deltaWeight < -0.5:
                trainLabel = -2
            elif -0.5 <= deltaWeight < 0:
                trainLabel = -1
            elif 0 <= deltaWeight <= 0.5:
                trainLabel = 0
            elif 0.5 < deltaWeight <= 1:
                trainLabel = 1
            elif 1 < deltaWeight <= 1.5:
                trainLabel = 2
            else:
                trainLabel = 3

        trainDatas.append(trainData)
        trainLabels.append(trainLabel)

    return (trainDatas, trainLabels)




trainDataList = processData(filePath='F:/up/t1/data/train.csv')

(trainDatas, trainLabels) = generateDataAndLabel(type='weight', metaDatas=trainDataList, week=1)

trainDatas = np.array(trainDatas)
trainLabels = np.array(trainLabels)

knn = neighbors.KNeighborsClassifier()
# 训练数据集
knn.fit(trainDatas, trainLabels)
# 预测




testDataList = processData(filePath='F:/up/t1/data/test.csv')

total_test_person = len(testDataList)
accurent =0

for onePerson in testDataList:
    #print (onePerson)
    test_list = []

    test_list.append(onePerson['weight0'])
    test_list.append(onePerson['height0'])
    test_list.append(onePerson['waistline0'])

    test_data = np.array([test_list])
    predict = knn.predict(test_data)

    predict_del = predict[0]
    predict_weight = onePerson['weight0'] - predict_del/2

    #predict_weight += random.uniform(0.2, 0.5)

    if abs(predict_weight - onePerson['weight4']) < 1.2:
        accurent += 1


print ("accurent = ", accurent / total_test_person * 100)