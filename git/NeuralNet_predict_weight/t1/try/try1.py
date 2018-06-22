
def processData(filePath):

    dataList = []

    with open(filePath, 'r') as f:
        records = f.readlines()

        print (records)
        print (len(records))

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

trainDataList = processData(filePath='/home/shen/Trying/Predict/up/t1/try/test.csv')
#trainDataList = processData(filePath='/home/shen/Trying/Predict/up/t1/data/test.csv')
