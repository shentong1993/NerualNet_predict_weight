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

            # if i ==2:
            #     print (dataList)

    return dataList



testDataList = processData(filePath='/home/shen/Trying/Predict/up/t1/data/trainData.csv')

dic ={'0-0.5':0,
      '0.5-1':0,
      '1-1.5':0,
      '1.5-2':0,
      '2-2.5':0,
      '2.5-3':0,

      '3-3.5':0,
      '3.5-4':0,
      '4-4.5':0,
      '4.5-5':0,
      '5-5.5':0,
      '5.5-6':0,

      '6-6.5': 0,
      '6.5-7': 0,
      '7-7.5': 0,
      '7.5-8': 0,
      '8-8.5': 0,
      '8.5-9': 0,
      '9-9.5': 0,
      '9.5-10': 0,
      }

deltaWeightAll = []
save_or_not = []
save_or_not_i =0
for onePerson in testDataList:
    deltaWeightAll.append( onePerson['deltaWaistAll'])
    if onePerson['deltaWeightAll']>= 0 and onePerson['deltaWeightAll'] <= 0.5:
        dic['0-0.5'] += 1
    elif onePerson['deltaWeightAll']>= 0.5 and onePerson['deltaWeightAll'] <= 1:
        dic['0.5-1'] += 1
    elif onePerson['deltaWeightAll']>= 1 and onePerson['deltaWeightAll'] <= 1.5:
        dic['1-1.5'] += 1
    elif onePerson['deltaWeightAll']>= 1.5 and onePerson['deltaWeightAll'] <= 2:
        dic['1.5-2'] += 1
    elif onePerson['deltaWeightAll']>= 2 and onePerson['deltaWeightAll'] <= 2.5:
        dic['2-2.5'] += 1
    elif onePerson['deltaWeightAll']>= 2.5 and onePerson['deltaWeightAll'] <= 3:
        dic['2.5-3'] += 1
    elif onePerson['deltaWeightAll']>= 3 and onePerson['deltaWeightAll'] <= 3.5:
        dic['3-3.5'] += 1
    elif onePerson['deltaWeightAll']>= 3.5 and onePerson['deltaWeightAll'] <= 4:
        dic['3.5-4'] += 1
    elif onePerson['deltaWeightAll']>= 4 and onePerson['deltaWeightAll'] <= 4.5:
        dic['4-4.5'] += 1


    elif onePerson['deltaWeightAll']>= 4.5 and onePerson['deltaWeightAll'] <= 5:
        dic['4.5-5'] += 1
    elif onePerson['deltaWeightAll']>= 5 and onePerson['deltaWeightAll'] <= 5.5:
        dic['5-5.5'] += 1
    elif onePerson['deltaWeightAll']>= 5.5 and onePerson['deltaWeightAll'] <= 6:
        dic['5.5-6'] += 1
    elif onePerson['deltaWeightAll']>= 6 and onePerson['deltaWeightAll'] <= 6.5:
        dic['6-6.5'] += 1
    elif onePerson['deltaWeightAll']>= 6.5 and onePerson['deltaWeightAll'] <= 7:
        dic['6.5-7'] += 1
    elif onePerson['deltaWeightAll']>= 7 and onePerson['deltaWeightAll'] <= 7.5:
        dic['7-7.5'] += 1
    elif onePerson['deltaWeightAll']>= 7.5 and onePerson['deltaWeightAll'] <= 8:
        dic['7.5-8'] += 1
    elif onePerson['deltaWeightAll']>= 8 and onePerson['deltaWeightAll'] <= 8.5:
        dic['8-8.5'] += 1

    elif onePerson['deltaWeightAll']>= 8.5 and onePerson['deltaWeightAll'] <= 9:
        dic['8.5-9'] += 1
    elif onePerson['deltaWeightAll']>= 9 and onePerson['deltaWeightAll'] <= 9.5:
        dic['9-9.5'] += 1
    elif onePerson['deltaWeightAll']>= 9.5 and onePerson['deltaWeightAll'] <= 10:
        dic['9.5-10'] += 1

    if onePerson['deltaWeightAll']<= 7:
        save_or_not.append(1)
    else:
        save_or_not.append(0)


print (deltaWeightAll)
print (len(deltaWeightAll))
print (max(deltaWeightAll))
print (min(deltaWeightAll))
print (dic)


with open("/home/shen/Trying/Predict/up/t1/data3_for_work/trainData.csv", "w", encoding='utf8') as csvfile:
    writer = csv.writer(csvfile)

        # 先写入columns_name
    writer.writerow([
            'checkinCount1', 'checkinCount2', 'checkinCount3', 'checkinCount4',
            'height0', 'weight0', 'weight1', 'weight2', 'weight3', 'weight4',
            'waistline0', 'waistline1', 'waistline2', 'waistline3', 'waistline4',
            'deltaWeight1', 'deltaWeight2', 'deltaWeight3', 'deltaWeight4', 'deltaWeightAll',
            'deltaWaist1', 'deltaWaist2', 'deltaWaist3', 'deltaWaist4', 'deltaWaistAll'])

    for i in testDataList:
        if save_or_not[save_or_not_i] ==1:
            writer.writerow([
            i['checkinCount1'], i['checkinCount2'], i['checkinCount3'], i['checkinCount4'],
            i['height0'], i['weight0'], i['weight1'], i['weight2'], i['weight3'], i['weight4'],
            i['waistline0'], i['waistline1'], i['waistline2'], i['waistline3'], i['waistline4'],
            i['deltaWeight1'], i['deltaWeight2'], i['deltaWeight3'], i['deltaWeight4'], i['deltaWeightAll'],
            i['deltaWaist1'], i['deltaWaist2'], i['deltaWaist3'], i['deltaWaist4'], i['deltaWaistAll']
            ])

        save_or_not_i += 1
