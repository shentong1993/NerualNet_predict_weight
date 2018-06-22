# import numpy as np
#
# n = np.array([[10, 20,30],
#               [11,22,30],
#               [13,22,30]
#               ])
#
# sub = np.array([[1],
#                 [2],
#                 [3]])
# print (n)
#
# n1 = [n[:, 0]]
# n2 = np.transpose(n1)
#
# print (n1)
#
# print (n2)
#
#
# new = sub /n2
#
# print (new)
#
# mean = np.mean(new)
# print ("mean = ",mean)


from PIL import Image
from pylab import *



# # 随意给的一些点
# x = [100, 100, 400, 400]
# y = [200, 500, 200, 500]

# 使用红色-星状标记需要绘制的点
plot(100, 200, 'r*')

# # 将数组中的前两个点进行连线
# plot(x[:2], y[:2])

# 添加标题信息
title('Plotting: "empire.jpg"')

# 隐藏坐标轴
# axis('off')

# 显示到屏幕窗口
show()