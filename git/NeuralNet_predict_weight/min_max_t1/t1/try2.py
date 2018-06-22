import numpy as np

testData =[45,34,36]

testData_array = np.array(testData)
print ("testData_array = ", testData_array)
print(type(testData_array))
print ("")

testData_array = testData_array[np.newaxis ,:]
print ("testData_array = ", testData_array)
print(type(testData_array))
print ("")

testData_array = testData_array.tolist()
print ("testData_array = ", testData_array)
print(type(testData_array))
print ("")


testData_array= testData_array[0]
print ("testData_array = ", testData_array)
print(type(testData_array))
print ("")