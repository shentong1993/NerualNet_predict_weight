import csv
import numpy as np


def processData(filePath):

    dataList = []

    with open(filePath, 'r') as f:
        records = f.readlines()

        # print (type(records))
        # print (records[0])
        #
        # print (type(records[0]))
        # r = records[0].strip()
        # print (type(r))
        # print (r)
        # print (records[1])

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

#[{The information of a person},{}]
trainDataList = processData(filePath='/home/shen/Trying/Predict/up/t1/more_dimension/more_dimension_data/trainData.csv')

len =  len(trainDataList)
print ("len = ", len)



delta_weight = []

for person in trainDataList:
    delta = person['weight0'] - person['weight4']

    delta_weight.append(delta)


delta_weight = np.array(delta_weight)
print (max(delta_weight))
print (min(delta_weight))


# num = 0
# right = 0
# for person in trainDataList:
#     delta = person['weight0'] - person['weight4']
#     if delta >= 0:
#         right += 1
#
# print (right)
# print (len)


# # make train.csv
# with open("/home/shen/Trying/Predict/up/t1/more_dimension/more_dimension_data/train.csv", "w", encoding='utf8') as csvfile:
#     writer = csv.writer(csvfile)
#
#     print ("hello")
#
#     # 先写入columns_name
#     writer.writerow(['checkinCount1','checkinCount2','checkinCount3','checkinCount4','height0','weight0',
#                      'weight1','weight2','weight3','weight4','waistline0','waistline1','waistline2'
#                      ,'waistline3','waistline4','deltaWeight1','deltaWeight2','deltaWeight3',
#                      'deltaWeight4','deltaWeightAll','deltaWaist1','deltaWaist2','deltaWaist3',
#                      'deltaWaist4','deltaWaistAll','hipline0','hipline1','hipline2','hipline3',
#                      'hipline4','chest0','chest1','chest2','chest3','chest4','thigh0','thigh1',
#                      'thigh2','thigh3','thigh4','arm0','arm1','arm2','arm3','arm4'
#                      ])
#
#
#     num = 0
#     for i in trainDataList:
#
#         if num < 4500 :
#             writer.writerow([
#                 i['checkinCount1'], i['checkinCount2'], i['checkinCount3'], i['checkinCount4'], i['height0'], i['weight0'],
#                 i['weight1'], i['weight2'], i['weight3'], i['weight4'], i['waistline0'], i['waistline1'], i['waistline2']
#                 , i['waistline3'], i['waistline4'], i['deltaWeight1'], i['deltaWeight2'], i['deltaWeight3'],
#                 i['deltaWeight4'], i['deltaWeightAll'], i['deltaWaist1'], i['deltaWaist2'], i['deltaWaist3'],
#                 i['deltaWaist4'], i['deltaWaistAll'], i['hipline0'], i['hipline1'], i['hipline2'], i['hipline3'],
#                 i['hipline4'], i['chest0'], i['chest1'], i['chest2'], i['chest3'], i['chest4'], i['thigh0'], i['thigh1'],
#                 i['thigh2'], i['thigh3'], i['thigh4'], i['arm0'], i['arm1'], i['arm2'], i['arm3'], i['arm4']
#
#             ])
#         num += 1
#
#
# # make test.csv
# with open("/home/shen/Trying/Predict/up/t1/more_dimension/more_dimension_data/test.csv", "w", encoding='utf8') as csvfile:
#     writer = csv.writer(csvfile)
#
#     print ("hello")
#
#     # 先写入columns_name
#     writer.writerow(['checkinCount1','checkinCount2','checkinCount3','checkinCount4','height0','weight0',
#                      'weight1','weight2','weight3','weight4','waistline0','waistline1','waistline2'
#                      ,'waistline3','waistline4','deltaWeight1','deltaWeight2','deltaWeight3',
#                      'deltaWeight4','deltaWeightAll','deltaWaist1','deltaWaist2','deltaWaist3',
#                      'deltaWaist4','deltaWaistAll','hipline0','hipline1','hipline2','hipline3',
#                      'hipline4','chest0','chest1','chest2','chest3','chest4','thigh0','thigh1',
#                      'thigh2','thigh3','thigh4','arm0','arm1','arm2','arm3','arm4'])
#
#     num = 0
#     for i in trainDataList:
#
#         if (num >=4500) and (num <= (len-1)) :
#             writer.writerow([
#                 i['checkinCount1'], i['checkinCount2'], i['checkinCount3'], i['checkinCount4'], i['height0'],
#                 i['weight0'],i['weight1'], i['weight2'], i['weight3'], i['weight4'], i['waistline0'], i['waistline1'],
#                 i['waistline2'], i['waistline3'], i['waistline4'], i['deltaWeight1'], i['deltaWeight2'], i['deltaWeight3'],
#                 i['deltaWeight4'], i['deltaWeightAll'], i['deltaWaist1'], i['deltaWaist2'], i['deltaWaist3'],
#                 i['deltaWaist4'], i['deltaWaistAll'], i['hipline0'], i['hipline1'], i['hipline2'], i['hipline3'],
#                 i['hipline4'], i['chest0'], i['chest1'], i['chest2'], i['chest3'], i['chest4'], i['thigh0'],
#                 i['thigh1'],i['thigh2'], i['thigh3'], i['thigh4'], i['arm0'], i['arm1'], i['arm2'], i['arm3'], i['arm4']
#             ])
#         num += 1