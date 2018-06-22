import csv
import numpy as np

def processData(filePath):

    dataList = []

    with open(filePath, 'r') as f:
        records = f.readlines()

        # 读第一行获取每一列的列名
        # strip() 去掉行尾的换行符
        keys = records[0].strip().split(',')
        print (keys)

        for i, record in enumerate(records):
            if i > 0:
                dic = {}
                values = record.strip().split(',')
                for index, key in enumerate(keys):
                    dic[key] = float(values[index])

                dataList.append(dic)

    return dataList

DataList = processData(filePath='/home/shen/Trying/Predict/up/t1/globle_minimal_question/data/trainData.csv')

save_list = []
len = len( DataList )
#len = 10
#print (len)

cut_num =10

num_of_each_cut = round(len / cut_num)
#print (num_of_each_cut)
n = np.arange(len)
print (n)



for i in range(cut_num - 1):
    temp = np.random.choice(n, num_of_each_cut, replace=False)
    #print (type(temp))
    n = np.array( list(set(n) - set(temp)) )

    temp = list(temp)
    save_list.append(temp)

save_list.append(list(n))

for i in range(cut_num ):
    print (save_list[i])
    # temp = save_list[i]
    # print (len(temp))



# save_list = [i=0[ j=0, , ,] ,[],[]]
for i in range(cut_num):

    with open("/home/shen/Trying/Predict/up/t1/globle_minimal_question/data_cut/data_cut%d/train.csv" % i, "w",
              encoding='utf8') as csvfile:
        writer = csv.writer(csvfile)

        # 先写入columns_name
        writer.writerow([
            'checkinCount1', 'checkinCount2', 'checkinCount3', 'checkinCount4',
            'height0', 'weight0', 'weight1', 'weight2', 'weight3', 'weight4',
            'waistline0', 'waistline1', 'waistline2', 'waistline3', 'waistline4',
            'deltaWeight1', 'deltaWeight2', 'deltaWeight3', 'deltaWeight4', 'deltaWeightAll',
            'deltaWaist1', 'deltaWaist2', 'deltaWaist3', 'deltaWaist4', 'deltaWaistAll'])

        for j in save_list[i]:
            writer.writerow([
                DataList[j]['checkinCount1'], DataList[j]['checkinCount2'], DataList[j]['checkinCount3'],
                DataList[j]['checkinCount4'],
                DataList[j]['height0'], DataList[j]['weight0'], DataList[j]['weight1'], DataList[j]['weight2'],
                DataList[j]['weight3'], DataList[j]['weight4'],
                DataList[j]['waistline0'], DataList[j]['waistline1'], DataList[j]['waistline2'],
                DataList[j]['waistline3'], DataList[j]['waistline4'],
                DataList[j]['deltaWeight1'], DataList[j]['deltaWeight2'], DataList[j]['deltaWeight3'],
                DataList[j]['deltaWeight4'], DataList[j]['deltaWeightAll'],
                DataList[j]['deltaWaist1'], DataList[j]['deltaWaist2'], DataList[j]['deltaWaist3'],
                DataList[j]['deltaWaist4'], DataList[j]['deltaWaistAll']
            ])


#write test
for i in range(cut_num):
    with open("/home/shen/Trying/Predict/up/t1/globle_minimal_question/data_cut/data_cut%d/test.csv" % i, "w",encoding='utf8') as csvfile:
        writer = csv.writer(csvfile)
            # 先写入columns_name
        writer.writerow([
                'checkinCount1', 'checkinCount2', 'checkinCount3', 'checkinCount4',
                'height0', 'weight0', 'weight1', 'weight2', 'weight3', 'weight4',
                'waistline0', 'waistline1', 'waistline2', 'waistline3', 'waistline4',
                'deltaWeight1', 'deltaWeight2', 'deltaWeight3', 'deltaWeight4', 'deltaWeightAll',
                'deltaWaist1', 'deltaWaist2', 'deltaWaist3', 'deltaWaist4', 'deltaWaistAll'])

        for k in range(cut_num):
            if k != i:
                for j in save_list[k]:
                    writer.writerow([
                        DataList[j]['checkinCount1'], DataList[j]['checkinCount2'], DataList[j]['checkinCount3'],
                        DataList[j]['checkinCount4'],
                        DataList[j]['height0'], DataList[j]['weight0'], DataList[j]['weight1'], DataList[j]['weight2'],
                        DataList[j]['weight3'], DataList[j]['weight4'],
                        DataList[j]['waistline0'], DataList[j]['waistline1'], DataList[j]['waistline2'],
                        DataList[j]['waistline3'], DataList[j]['waistline4'],
                        DataList[j]['deltaWeight1'], DataList[j]['deltaWeight2'], DataList[j]['deltaWeight3'],
                        DataList[j]['deltaWeight4'], DataList[j]['deltaWeightAll'],
                        DataList[j]['deltaWaist1'], DataList[j]['deltaWaist2'], DataList[j]['deltaWaist3'],
                        DataList[j]['deltaWaist4'], DataList[j]['deltaWaistAll']
                    ])







