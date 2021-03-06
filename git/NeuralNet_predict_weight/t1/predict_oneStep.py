from sklearn import linear_model

import os
import random

c =0
sum = 0

dicbim={
    '0-20':  0,
    '20-23': 0,
    '23-26': 0,
    '26-30': 0,
    '30-35': 0,
    '35-100':0
}
FEATURESMAP = {
    1: ['weight0', 'height0', 'waistline0', 'checkinCount1'],
    2: ['weight1', 'height0', 'waistline1', 'checkinCount2'],
    3: ['weight2', 'height0', 'waistline2', 'checkinCount3'],
    4: ['weight3', 'height0', 'waistline3', 'checkinCount4']
}

WEIGHTLABELSMAP = {
    1: 'deltaWeightAll',
    2: 'deltaWeight2',
    3: 'deltaWeight3',
    4: 'deltaWeight4'
}

WAISTLABELSMAP = {
    1: 'deltaWaistAll',
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


    return dataList



def generateTestData(metaData, week):

    testDataDic = {}
    height = metaData['height0']
    weight = metaData['weight%d' %(week-1)]
    waist = metaData['waistline%d' %(week-1)]
    testDataDic['originWeight'] = weight
    testDataDic['originWaist'] = waist
    testDataDic['deltaWeightAll'] = metaData['deltaWeightAll']

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



    # if 20 < bmi <= 23:
    #     predictResDelta = 5

    # if not ( (predictResDelta/2) <= predictableData['deltaWeightAll']  <= ((predictResDelta/2)+0.5) ):
    #     global  c
    #     c += 1
    #
    #     print ('bmi = ',bmi)
    #     print ("predictResDelta = ", predictResDelta/2)
    #     print ("true = ", predictableData['deltaWeightAll'])
    #     print ("c = ",c)
    #     print (" ")

    # if bmi > 28 and predictResDelta <= 3:
    #     predictResDelta += 1
    # if bmi < 21 and predictResDelta >= 1:
    #     if bmi < 20:
    #         predictResDelta = 0
    #     else:
    #         predictResDelta -= 1


    offset = (bmi - BMIMAP[bodyLevel][0]) / (
    BMIMAP[bodyLevel][1] - BMIMAP[bodyLevel][0])

    # if offset < 0:
    #     pass
    # elif 0 < offset <= 0.1:
    #     offset += random.uniform(0.1, 0.2)
    # elif 0.1 < offset <= 0.2:
    #     offset += random.uniform(0.08, 0.13)
    # elif 0.2 < offset < 0.5:
    #     offset += random.uniform(0.05, 0.1)
    # elif 1 < offset <= 1.3:
    #     offset -= random.uniform(0.08, 0.13)
    # elif 1.3 < offset <= 1.5:
    #     offset -= random.uniform(0.1, 0.2)
    # else:
    #     offset -= random.uniform(0.2, 0.4)



    # if offset < 0:
    #     pass
    # elif 0 < offset <= 0.1:
    #     offset += random.uniform(0.2, 0.4)
    # elif 0.1 < offset <= 0.2:
    #     offset += random.uniform(0.3, 0.4)
    # elif 0.2 < offset < 0.5:
    #     offset += random.uniform(0.1, 0.2)
    #
    #
    # # #change
    # elif 0.5 <= offset <= 0.8:
    #     offset -= random.uniform(0.1, 0.2)
    #
    # elif 0.8 <= offset <= 0.9:
    #     offset -= random.uniform(0.3, 0.4)
    #
    # else:
    #     offset -= random.uniform(0.3 ,0.4)


    predictResDelta = float(predictResDelta / 2) #+ offset

    if type == 'weight':
        predictRes = originWeight - predictResDelta

    elif type == 'waist':
        predictRes = originWaist - predictResDelta + random.uniform(-0.1, 0.1)

   #weight4 = originWeight - predictableData['deltaWeightAll']

    if not ((predictResDelta / 2) <= predictableData['deltaWeightAll'] <= ((predictResDelta / 2) + 0.5)):
    #if abs(weight4 - predictRes ) >=1.2:
        global  c
        global dicbim
        global sum

        if 0 < bmi <= 20:
            dicbim['0-20'] +=1
        elif 20 < bmi <= 23:
            dicbim['20-23'] +=1
        elif 23 < bmi <= 26:
            dicbim['23-26'] +=1
        elif 26 < bmi <= 30:
            dicbim['26-30'] +=1
        elif 30 < bmi <= 35:
            dicbim['30-35'] +=1
        else:
            dicbim['35-100'] +=1


        if 20 < bmi <= 23:

            print('bmi = ', bmi)
            print("predictResDelta = ", predictResDelta / 2)
            print("true = ", predictableData['deltaWeightAll'])
            print("c = ", c)
            print(" ")
            #if predictableData['deltaWeightAll'] >3:
            sum += predictableData['deltaWeightAll']
            c += 1


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

            else:
                trainLabel = 11
            # elif 5.5 <= deltaWeight <= 6:
            #     trainLabel = 11
            # elif 6 <= deltaWeight <= 6.5:
            #     trainLabel = 12
            #
            # elif 6.5 <= deltaWeight <= 7:
            #     trainLabel = 13
            # elif 7 <= deltaWeight <= 7.5:
            #     trainLabel = 14
            # elif 7.5 <= deltaWeight < 8:
            #     trainLabel = 15
            #
            # elif 8 <= deltaWeight <= 8.5:
            #     trainLabel = 16
            # elif 8.5 <= deltaWeight <= 9:
            #     trainLabel = 17
            # elif 9 <= deltaWeight <= 9.5:
            #     trainLabel = 18
            # else:
            #     trainLabel =19

            # else:
            #     trainLabel =16
            #
            # else:
            #     trainLabel =13


            #






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

trainDataList = processData(filePath='F:/up/t1/data/train.csv')
weightClf1 = generateClf(type='weight', metaDatas=trainDataList, week=1)
weightClf2 = generateClf(type='weight', metaDatas=trainDataList, week=2)
weightClf3 = generateClf(type='weight', metaDatas=trainDataList, week=3)
weightClf4 = generateClf(type='weight', metaDatas=trainDataList, week=4)
waistClf1 = generateClf(type='waist', metaDatas=trainDataList, week=1)
waistClf2 = generateClf(type='waist', metaDatas=trainDataList, week=2)
waistClf3 = generateClf(type='waist', metaDatas=trainDataList, week=3)
waistClf4 = generateClf(type='waist', metaDatas=trainDataList, week=4)

#[{ all values of a person },{}...]
testDataList = processData(filePath='F:/up/t1/data/test.csv')


result = []
accurent =0
accurent2 =0
count =0

total = len(testDataList)

for onePerson in testDataList:

    dic = {}

    week =  1
    dataDic = generateTestData(metaData=onePerson, week=week)
    predictWeight = predict(type='weight', predictableData=dataDic, clf=getClf(type='weight', week=week))
    #predictWaist = predict(type='waist', predictableData=dataDic, clf=getClf(type='waist', week=week))

    dic['weight4'] = predictWeight
    #dic['waistline4'] = predictWaist

    if predictWeight <  onePerson['weight0'] :
        count += 1

    if abs(predictWeight - onePerson['weight4']) < 0.8:
        accurent += 1
    #if abs(predictWaist - onePerson['waistline4']) < 3:
    #    accurent2 += 1

    result.append(dic)

print ("result = ",result)

#last step the number that loss weight
print ("right number = ", count)
print ("total number = ",total)
print ("accurent = ", accurent/total * 100)
#print ("accurent = ", accurent2/total * 100)

print ('c = ',c)
print (dicbim)
#sum = sum / c
print("average True = ", sum)



