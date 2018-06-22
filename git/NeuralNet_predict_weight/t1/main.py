LABELMAP = {}
LABELMAP = {
    1: 'deltaWeight1',
    2: 'deltaWeight2',
    3: 'deltaWeight3',
    4: 'deltaWeight4'
}

FEATURESMAP = {
    1: ['weight0', 'height0', 'waistline0', 'checkinCount1'],
    2: ['weight1', 'height0', 'waistline1', 'checkinCount2'],
    3: ['weight2', 'height0', 'waistline2', 'checkinCount3'],
    4: ['weight3', 'height0', 'waistline3', 'checkinCount4']
}
#print (LABELMAP)

def processData(filePath):

    dataList = []

    with open(filePath, 'r') as f:
        records = f.readlines()
        print ("records = ", type(records))


        # 读第一行获取每一列的列名
        # strip() 去掉行尾的换行符
        keys = records[0].strip().split(',')
        print ("key ", keys)

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

        #print ("trainData = ",trainData )


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

        #print ("trainDatas = ",trainDatas)
        print ("trainLabels = ",trainLabels)

    return (trainDatas, trainLabels)



def generateClf(type, metaDatas, week):

    (trainDatas, trainLabels) = generateDataAndLabel(type=type, metaDatas=metaDatas, week=week)
    return


trainDataList = processData('/Users/shen/PycharmProjects/t1/trainData.csv')

generateClf(type='weight', metaDatas=trainDataList, week=3)

# print (trainDataList)
# print (type(trainDataList))
# print ("\n" *3)



#
#
# for item in trainDataList:
#     print (item)
#     print (type(item))