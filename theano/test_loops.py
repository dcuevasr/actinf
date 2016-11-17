#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 12:26:30 2016

@author: dario
"""

import theano as th
import theano.tensor as T
import numpy as np

X = T.matrix('X')
Y = T.matrix('Y')

def fboring(x,y):
    return x**y

results, updates = th.scan(fboring, sequences=[X, Y])
f_th = th.function(inputs=[X, Y], outputs=results)

x = np.reshape(np.arange(4, dtype = th.config.floatX), (2,2))
y = np.reshape(np.arange(4,8, dtype = th.config.floatX), (2,2))
w = np.ones((2,2), dtype = th.config.floatX)
m = 2*np.ones((2,2), dtype = th.config.floatX)

print(f_th(x, y))

#%%

a = th.shared(1)
values, updates = th.scan(lambda: {a: a+1}, n_steps = 5)

b = a + 1
c = updates[a] + 1
f_2 = th.function([], [b,c], updates=updates)

print(a.get_value())
print(b.get_value())
print(c.get_value())
