# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 17:44:58 2016

@author: dario
"""

import numpy as np
import os
import sys


def yell():
    print('yo')


class cd:
    """ Context for temporarily changing working folder. """

    def __init__(self, newPath, force=False):
        self.newPath = newPath
        self.force = force

    def __enter__(self):
        self.oldPath = os.getcwd()
        try:
            os.chdir(self.newPath)
        except FileNotFoundError:
            if self.force:
                os.mkdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.oldPath)


class shutup:
    """ Context for suppressing output from functions. It temporarily sets
    stdout and stderr to empty in-memory files.

    Optionally, it can be set to write them to a text file.
    """

    def __init__(self, out_file=None, verbose=True):
        """ If out_file is provided and can be written to, it will write all
        output to that file.
        """
        if out_file is not None:
            try:
                mafi = open(out_file, 'w')
#                osfi = os.open(out_file, os.O_RDWR, os.O_CREAT)
#                self.temp_files = [osfi, osfi]
                self.out_null = mafi
            except:
                #                raise
                out_file = None
        if out_file is None:
            self.temp_files = [os.open(os.devnull, os.O_RDWR)
                               for x in range(2)]
            self.out_null = open(os.devnull, 'w')
        self.save_dup = (os.dup(1), os.dup(2))

    def __enter__(self):
        self.orig_out = sys.stdout
        sys.stdout = self.out_null
        os.dup2(self.temp_files[0], 1)
        os.dup2(self.temp_files[1], 2)

    def __exit__(self, etype, value, traceback):
        sys.stdout = self.orig_out
        os.dup2(self.save_dup[0], 1)
        os.dup2(self.save_dup[1], 2)
        # close the temp files
        os.close(self.temp_files[0])
        os.close(self.temp_files[1])
        self.out_null.close()


def calc_subplots(nSubjects):
    """ Calculates a good arrangement for the subplots given the number of
    subjects.
    # TODO: Make it smarter for prime numbers
    """
    if nSubjects == 2 or nSubjects == 3:
        return nSubjects, 1

    sqrtns = nSubjects**(1 / 2)
    if abs(sqrtns - np.ceil(sqrtns)) < 0.001:
        a1 = a2 = np.ceil(sqrtns)
    else:
        divs = nSubjects % np.arange(2, nSubjects)
        divs = np.arange(2, nSubjects)[divs == 0]
        if divs.size == 0:
            a1 = np.ceil(nSubjects / 2)
            a2 = 2
        else:
            a1 = divs[np.ceil(len(divs) / 2).astype(int)]
            a2 = nSubjects / a1
    return int(a1), int(a2)


def ismember(A, B):
    return np.array([np.sum(a == B) for a in A]).astype(bool)


def softmax(A):

    #    if type(A) is not np.ndarray:
    #        raise Exception('Only numpy.nparray are accepted')
    #    else:
    #        if np.prod(np.shape(A)) != np.shape(A)[0]:
    #            raise Exception('Only vectors are accepted!')
    maxA = A.max()
    A = A - maxA
    A = np.exp(A)
    A = A / np.sum(A)
    return A


def allothers(Indices, Dimensions):
    """
     Using np.ravel_multi_index, return an np.array of all the linear indices
     pertaining all the possible combinations of indices in Indices, assuming
     an array of dimensions specified in Dimensions.

     Given that it uses three for-loops, it is highly inefficient. Use only
     for small sets of indices.
    """
    if len(Indices) == 2:
        Indices.append([0])
    if len(Indices) != 3:
        raise Exception('Waaa?')

    if len(Dimensions) == 2:
        Dimensions = Dimensions + (1,)
    Iout = []
    for x in range(len(Indices[0])):
        for y in range(len(Indices[1])):
            for z in range(len(Indices[2])):
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
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    newFunc.__name__ = func.__name__
    newFunc.__doc__ = func.__doc__
    newFunc.__dict__.update(func.__dict__)
    return newFunc
