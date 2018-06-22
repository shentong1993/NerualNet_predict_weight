import numpy as np

trainDatas = [[10,20,30],
              [30,40,50],
              [50,60,70]]
print (type(trainDatas))
print ("")

trainDatas = np.array(trainDatas)
print (type(trainDatas))
print ("")

trainDatas = trainDatas.tolist()
print (type(trainDatas))
print ("")