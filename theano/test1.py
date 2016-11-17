# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import theano as th
import numpy as np

x = th.tensor.dmatrix('x')
y = th.tensor.dmatrix('y')
z = th.tensor.dmatrix('z')

xy = x + y
xz = x + z
yz = y + z

s = xy - xz

s2 = 2*s

s3 = x + y + z

m1 = x*y
m2 = x*z
m3 = y*z

sample1= th.function([x,y], xy)
sample2 = th.function([x,y,z], [s2, s3])
sample3 = th.function([x,y,z], [m1, m2, m3])
xin = np.reshape(np.arange(4), (2,2)) 
yin = np.reshape(np.arange(1,5), (2,2))
zin = np.reshape(np.arange(2,6), (2,2))

print(sample1(xin, yin))
print(sample2(xin, yin, zin))
print(sample3(xin, yin, zin))
