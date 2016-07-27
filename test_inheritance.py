# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:45:23 2016

@author: dario
"""

class mparent(object):
    def __init__(self, name):
        print 'hello world from Parent'
        self.name = name
    def cosa(self):
        print 'my name is: %s' % self.name
        
        
class mchild(mparent):
    def __init__(self,name):
        print 'hello world from Child'
        self.name = name
    def cosita(self):
        self.cosa()
        