'''
@author: lyx
@contact: woshiluyangxing@qq.com
@file: showimg.py
@time: 2019-12-05 19:22
@desc:
'''
import matplotlib.pyplot as plt
import numpy as np
s=np.load("score760000.npy")
lenth=len(s)
# plt.hist(s,np.arange(5,20))
# plt.show()



mean1 = np.mean(s)

std_deviation1 = np.std(s)

plt.errorbar( s,mean1, std_deviation1, fmt="o")
plt.show()
