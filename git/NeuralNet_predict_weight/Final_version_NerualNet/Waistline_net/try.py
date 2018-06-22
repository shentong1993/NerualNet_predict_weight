import numpy as np

FEATURESMAP = {
    1: ['weight0', 'height0', 'waistline0'],
    #1: ['weight0', 'height0'],
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
    waistLineLabels = []

    useableKeys = FEATURESMAP[week]

    for metaData in metaDatas:
        trainData = []
        for key in useableKeys:
            trainData.append(metaData[key])
        deltaWeight = metaData[LABELMAP[week]]
        waistLine = metaData['waistline0']
        #waistLine = metaData['deltaWeightAll']

        trainDatas.append(trainData)
        trainLabels.append(deltaWeight)
        waistLineLabels.append(waistLine)

    return (trainDatas, trainLabels, waistLineLabels)


trainDataList = processData(filePath='/home/shen/Trying/Predict/up/t1/more_dimension/more_dimension_data/trainData.csv')

(trainDatas, trainLabels, waistLineLabels) = generateDataAndLabel(type='waist', metaDatas=trainDataList, week=1)

wl = np.array(waistLineLabels)

print (wl)
print (wl.shape)
print ("max = ",np.max(wl))
print ("min = ",np.min(wl))