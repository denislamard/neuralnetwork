# coding: utf-8
#!/usr/bin/env python3

import math
import numpy
import scipy.special
import datetime

def timetosecond(time):
    ftr = [3600,60,1]
    return sum([a*b for a,b in zip(ftr, map(int,time.split(':')))])  

def scaledata(value, max):
    return float('{0:.6f}'.format(value / (max * 1)))

def creatdata(data):
    tab = data.split(';')
    d = datetime.datetime.strptime(tab[0], '%d/%m/%Y').date()
    return [scaledata(d.weekday(), 7), scaledata(d.month, 12), scaledata(timetosecond(tab[1]), 85800)]

class NeuralNetwork():

    def __init__(self):
        self.inodes = 5
        self.hnodes = 10
        self.onodes = 2
        self.lr = 0.1
        
        self.wih = numpy.random.rand(self.hnodes, self.inodes)
        self.who = numpy.random.rand(self.onodes, self.hnodes)
        
        self.activation_function = lambda x: scipy.special.expit(x)
    
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
    
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
    
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
    
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors) 
    
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

    
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
    
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
 
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
    
        return final_outputs
    
if __name__ == '__main__':  
    
    neural = NeuralNetwork()
    for index in range(10000):
        neural.train([0.1, 0.2, 0.3, 0.9, 0.9], [0.999, 0.1])
        neural.train([0.4, 0.5, 0.6, 0.1, 0.7], [0.585, 0.9])
        neural.train([0.7, 0.1, 0.9, 0.2, 0.5], [0.104, 0.1])
        neural.train([0.7, 0.8, 0.9, 0.3, 0.3], [0.305, 0.9])
        neural.train([0.1, 0.8, 0.5, 0.7, 0.1], [0.680, 0.9])
        
    print ('Query=', [0.1, 0.2, 0.3, 0.9, 0.9])    
    ret = neural.query([0.1, 0.2, 0.3, 0.9, 0.9])
    for i in range (0, len(ret)):
        print ('{0:.6f}'.format(float(ret[i])))        