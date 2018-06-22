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



testDataList = processData(filePath='F:/up/min_max_t1/t1/data/train.csv')

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

print (deltaWeightAll)
print (len(deltaWeightAll))
print (max(deltaWeightAll))
print (min(deltaWeightAll))
print (dic)