from sklearn import linear_model

import os
import random

FEATURESMAP = {
    1: ['weight0', 'height0', 'waistline0', 'checkinCount1'],
    2: ['weight1', 'height0', 'waistline1', 'checkinCount2'],
    3: ['weight2', 'height0', 'waistline2', 'checkinCount3'],
    4: ['weight3', 'height0', 'waistline3', 'checkinCount4']
}

WEIGHTLABELSMAP = {
    1: 'deltaWeight1',
    2: 'deltaWeight2',
    3: 'deltaWeight3',
    4: 'deltaWeight4'
}

WAISTLABELSMAP = {
    1: 'deltaWaist1',
    2: 'deltaWaist2',
    3: 'deltaWaist3',
    4: 'deltaWaist4'
}

BMIMAP = {
    0: (0, 20),
    1: (20, 23),
    2: (23, 26),
    3: (26, 30),
    4: (30, 35),
    5: (35, 100)
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

            # if i ==2:
            #     print (dataList)

    return dataList



def generateTestData(metaData, week):

    testDataDic = {}
    height = metaData['height0']
    weight = metaData['weight%d' %(week-1)]
    waist = metaData['waistline%d' %(week-1)]
    testDataDic['originWeight'] = weight
    testDataDic['originWaist'] = waist

    bmi = round(weight / ((height / 100) * (height / 100)), 2)
    testDataDic['bmi'] = bmi
    if bmi <= 20:
        testDataDic['bodyLevel'] = 0
    elif 20 < bmi <= 23:
        testDataDic['bodyLevel'] = 1
    elif 23 < bmi <= 26:
        testDataDic['bodyLevel'] = 2
    elif 26 < bmi <= 30:
        testDataDic['bodyLevel'] = 3
    elif 30 < bmi <= 35:
        testDataDic['bodyLevel'] = 4
    else:
        testDataDic['bodyLevel'] = 5

    useableKeys = FEATURESMAP[week]

    testData = []
    for key in useableKeys:
        if key == 'checkinCount%s' % (week):
            value = 28
        else:
            value = metaData[key]
        testData.append(value)

    testDataDic['data'] = testData


    return testDataDic




def predict(type, predictableData, clf):

    X = [predictableData['data']]
    bmi = predictableData['bmi']
    bodyLevel = predictableData['bodyLevel']
    originWeight = predictableData['originWeight']
    originWaist = predictableData['originWaist']
    predictResDelta = clf.predict(X)


    if bmi > 28 and predictResDelta <= 3:
        predictResDelta += 1
    if bmi < 21 and predictResDelta >= 1:
        if bmi < 20:
            predictResDelta = 0
        else:
            predictResDelta -= 1
    offset = (bmi - BMIMAP[bodyLevel][0]) / (
    BMIMAP[bodyLevel][1] - BMIMAP[bodyLevel][0])

    if offset < 0:
        pass
    elif 0 < offset <= 0.1:
        offset += random.uniform(0.1, 0.2)
    elif 0.1 < offset <= 0.2:
        offset += random.uniform(0.08, 0.13)
    elif 0.2 < offset < 0.5:
        offset += random.uniform(0.05, 0.1)
    elif 1 < offset <= 1.3:
        offset -= random.uniform(0.08, 0.13)
    elif 1.3 < offset <= 1.5:
        offset -= random.uniform(0.1, 0.2)
    else:
        offset -= random.uniform(0.2, 0.4)

    predictResDelta = float(predictResDelta / 2) + offset


    # if predictResDelta < 0:
    #     predictResDelta = -predictResDelta

    if type == 'weight':
        predictRes = originWeight - predictResDelta
    elif type == 'waist':
        predictRes = originWaist - predictResDelta + random.uniform(-0.1, 0.1)

    # if predictRes < 1:
    #     predictRes += float(1)

    return round(predictRes, 2)





def getClf(type, week):
    if type == 'weight':
        if week == 1:
            return weightClf1
        if week == 2:
            return weightClf2
        if week == 3:
            return weightClf3
        if week == 4:
            return weightClf4
    if type == 'waist':
        if week == 1:
            return waistClf1
        if week == 2:
            return waistClf2
        if week == 3:
            return waistClf3
        if week == 4:
            return waistClf4






def generateDataAndLabel(type, metaDatas, week):
    LABELMAP = {}
    if type == 'weight':
        LABELMAP = {
            1: 'deltaWeight1',
            2: 'deltaWeight2',
            3: 'deltaWeight3',
            4: 'deltaWeight4'
        }
    elif type == 'waist':
        LABELMAP = {
            1: 'deltaWaist1',
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
            if deltaWeight < -0.5:
                trainLabel = -2
            elif -0.5 <= deltaWeight < 0:
                trainLabel = -1
            elif 0 <= deltaWeight <= 0.5:
                trainLabel = 0
            else:
                trainLabel = 1
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





def generateClf(type, metaDatas, week):

    (trainDatas, trainLabels) = generateDataAndLabel(type=type, metaDatas=metaDatas, week=week)
    clf = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', C=0.00003)
    clf.fit(trainDatas, trainLabels)

    return clf

trainDataList = processData(filePath='/Users/shen/PycharmProjects/t1/data/train.csv')
weightClf1 = generateClf(type='weight', metaDatas=trainDataList, week=1)
weightClf2 = generateClf(type='weight', metaDatas=trainDataList, week=2)
weightClf3 = generateClf(type='weight', metaDatas=trainDataList, week=3)
weightClf4 = generateClf(type='weight', metaDatas=trainDataList, week=4)
waistClf1 = generateClf(type='waist', metaDatas=trainDataList, week=1)
waistClf2 = generateClf(type='waist', metaDatas=trainDataList, week=2)
waistClf3 = generateClf(type='waist', metaDatas=trainDataList, week=3)
waistClf4 = generateClf(type='waist', metaDatas=trainDataList, week=4)

#[{ all values of a person },{}...]
testDataList = processData(filePath='/Users/shen/PycharmProjects/t1/data/test.csv')


result = []
count =0
accurent =0

accurent1 =0
accurent2 =0
accurent3 = 0
accurent4 = 0


weight_loss_number =0
total = len(testDataList)

for onePerson in testDataList:
    dic = {}

    if onePerson['deltaWeightAll'] >0:
        weight_loss_number += 1

    for i in range(4):
        week = i+1
        dataDic = generateTestData(metaData=onePerson, week=week)
        predictWeight = predict(type='weight', predictableData=dataDic, clf=getClf(type='weight', week=week))
        predictWaist = predict(type='waist', predictableData=dataDic, clf=getClf(type='waist', week=week))

        dic['weight%d' %week] = predictWeight
        dic['waistline%d' %week ] = predictWaist



        if week == 1 and abs(predictWeight - onePerson['weight1']) < 1.2:
            accurent1 += 1
        elif week == 2 and abs(predictWeight - onePerson['weight2']) < 1.2:
            accurent2 += 1
        elif week == 3 and abs(predictWeight - onePerson['weight3']) < 1.2:
            accurent3 += 1
        elif week == 4 and abs(predictWeight - onePerson['weight4']) < 1.2:
            accurent4 += 1



        if week != 4:

            onePerson["weight%d" %week] =   predictWeight
            onePerson["waistline%d" %week] = predictWaist




        if week == 4:

            preweek = week -1
            if predictWeight <=   onePerson['weight%d' %preweek]:
                count += 1

            if abs(predictWeight - onePerson['weight4']) < 1.2:
                accurent += 1
    result.append(dic)

print ("result = ",result)


#last step the number that loss weight
print ("right number = ", count)
print ("total number = ",total)
print ("weight_loss_number = ", weight_loss_number)
print ("accurent = ", accurent/total * 100)
print ("accurent1 = ", accurent1/total * 100)
print ("accurent2 = ", accurent2/total * 100)
print ("accurent3 = ", accurent3/total * 100)
print ("accurent4 = ", accurent4/total * 100)


