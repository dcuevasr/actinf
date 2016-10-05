# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 17:44:58 2016

@author: dario
"""

import numpy as np

def ismember(A,B):
    return np.array([np.sum(a == B) for a in A]).astype(bool)

def softmax(A):

    if type(A) is not np.ndarray:
        raise Exception('Only numpy.nparray are accepted')
    else:
        if np.prod(np.shape(A)) != np.shape(A)[0]:
            raise Exception('Only vectors are accepted!')
    maxA = A.max()
    A = A - maxA
    A = np.exp(A)
    A = A/np.sum(A)
    return A

def allothers(Indices, Dimensions):
    """
     Using np.ravel_multi_index, return an np.array of all the linear indices
     pertaining all the possible combinations of indices in Indices, assuming
     an array of dimensions specified in Dimensions.

     Given that it uses three for-loops, it is highly inefficient. Use only
     for small sets of indices.
    """
    if len(Indices)==2:
        Indices.append([0])
    if len(Indices)!=3:
        raise Exception('Waaa?')

    if len(Dimensions)==2:
        Dimensions = Dimensions + (1,)
    Iout = []
    for x in xrange(len(Indices[0])):
        for y in xrange(len(Indices[1])):
            for z in xrange(len(Indices[2])):
                Iout.append(np.ravel_multi_index([Indices[0][x],
                                                  Indices[1][y],
                                                  Indices[2][z]],
                                                  Dimensions, order='F'))
    return Iout




import warnings

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""
    def newFunc(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)#turn off filter
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    newFunc.__name__ = func.__name__
    newFunc.__doc__ = func.__doc__
    newFunc.__dict__.update(func.__dict__)
    return newFunc