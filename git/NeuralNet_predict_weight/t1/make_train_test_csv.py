import csv


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

            if i ==2:
                print (dataList)

    return dataList


trainDataList = processData(filePath='/home/shen/Trying/Predict/up/t1/data3_more_accurcy/trainData.csv')

len =  len(trainDataList)
print ("len = ", len)

# make train.csv
with open("/home/shen/Trying/Predict/up/t1/data3_more_accurcy/train.csv", "w", encoding='utf8') as csvfile:
    writer = csv.writer(csvfile)

    print ("hello")

    # 先写入columns_name
    writer.writerow([
        'checkinCount1', 'checkinCount2', 'checkinCount3', 'checkinCount4',
        'height0', 'weight0', 'weight1', 'weight2', 'weight3', 'weight4',
        'waistline0', 'waistline1', 'waistline2', 'waistline3', 'waistline4',
        'deltaWeight1', 'deltaWeight2', 'deltaWeight3', 'deltaWeight4', 'deltaWeightAll',
        'deltaWaist1', 'deltaWaist2', 'deltaWaist3', 'deltaWaist4', 'deltaWaistAll'])

    num = 0
    for i in trainDataList:

        if num <2300 :
            writer.writerow([
            i['checkinCount1'], i['checkinCount2'], i['checkinCount3'], i['checkinCount4'],
            i['height0'], i['weight0'], i['weight1'], i['weight2'], i['weight3'], i['weight4'],
            i['waistline0'], i['waistline1'], i['waistline2'], i['waistline3'], i['waistline4'],
            i['deltaWeight1'], i['deltaWeight2'], i['deltaWeight3'], i['deltaWeight4'], i['deltaWeightAll'],
            i['deltaWaist1'], i['deltaWaist2'], i['deltaWaist3'], i['deltaWaist4'], i['deltaWaistAll']
            ])
        num += 1


# make test.csv
with open("/home/shen/Trying/Predict/up/t1/data3_more_accurcy/test.csv", "w", encoding='utf8') as csvfile:
    writer = csv.writer(csvfile)

    print ("hello")

    # 先写入columns_name
    writer.writerow([
        'checkinCount1', 'checkinCount2', 'checkinCount3', 'checkinCount4',
        'height0', 'weight0', 'weight1', 'weight2', 'weight3', 'weight4',
        'waistline0', 'waistline1', 'waistline2', 'waistline3', 'waistline4',
        'deltaWeight1', 'deltaWeight2', 'deltaWeight3', 'deltaWeight4', 'deltaWeightAll',
        'deltaWaist1', 'deltaWaist2', 'deltaWaist3', 'deltaWaist4', 'deltaWaistAll'])

    num = 0
    for i in trainDataList:

        if (num >=2300) and (num <= (len-1)) :
            writer.writerow([
            i['checkinCount1'], i['checkinCount2'], i['checkinCount3'], i['checkinCount4'],
            i['height0'], i['weight0'], i['weight1'], i['weight2'], i['weight3'], i['weight4'],
            i['waistline0'], i['waistline1'], i['waistline2'], i['waistline3'], i['waistline4'],
            i['deltaWeight1'], i['deltaWeight2'], i['deltaWeight3'], i['deltaWeight4'], i['deltaWeightAll'],
            i['deltaWaist1'], i['deltaWaist2'], i['deltaWaist3'], i['deltaWaist4'], i['deltaWaistAll']
            ])
        num += 1