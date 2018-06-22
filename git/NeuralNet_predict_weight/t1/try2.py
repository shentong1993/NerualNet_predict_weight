# from PIL import Image
# from pylab import *
#
#
#
# x = [100, 100, 400, 400]
# y = [200, 500, 200, 500]
#
# for i in range(len(x)):
#     if i ==0:
#         plot(x[i], y[i], 'r*')
#     else :
#         plot(x[i], y[i], 'b*')
#
#
#
#
# title('Plotting: "empire.jpg"')
#
#
# show()

import numpy as np
from PIL import Image
from pylab import *

n =[[10],
    [20],
    [30],
    [40]]

n = np.array(n)
print (n)
print (n.shape)

n1 = n.transpose()
print (n1)

n2 = n1.squeeze()
print (n2)

y = n2

print (y.shape[0])

x = np.arange(y.shape[0])

print (x)


for i in range(len(x)):
    if i ==0:
        plot(x[i], y[i], 'r*')
    else :
        plot(x[i], y[i], 'b*')


show()