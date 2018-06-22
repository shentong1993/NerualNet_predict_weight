
def processData(filePath):

    dataList = []

    with open(filePath, 'r') as f:
        records = f.readlines()


        keys = records[0].strip().split(',')

        for i, record in enumerate(records):
            if i > 0:
                dic = {}
                values = record.strip().split(',')
                for index, key in enumerate(keys):
                    dic[key] = float(values[index])

                dataList.append(dic)

    return dataList

trainDataList = processData(filePath='/home/shen/Trying/Predict/up/t1/data/train.csv')

