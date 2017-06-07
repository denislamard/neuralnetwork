# coding: utf-8
#!/usr/bin/env python3

import math
import numpy
import scipy.special
import datetime

def timetosecond(time):
    time = time.replace(":", "")
    return float(time) / 10000 + 0.00001    
    
def scaledata(value, max):
    return float('{0:.2f}'.format(value / (max * 0.99)))

def creatdata(data):
    tab = data.split(';')
    d = datetime.datetime.strptime(tab[0], '%d/%m/%Y').date()
    return [(d.weekday()+1)/10, scaledata(timetosecond(tab[1]), 85800)]

class NeuralNetwork():

    def __init__(self):
        self.inodes = 2
        self.hnodes = 2
        self.onodes = 1
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
    
    result = []
    with open('/home/denis/neurones/data.txt', 'r') as f:
        for data in f:
            try:
                tab = data[:len(data)-1].split(';')
                d = datetime.datetime.strptime(tab[0], '%d/%m/%Y').date()
                result.append([(d.weekday()+1)/10, timetosecond(tab[1]), scaledata(int(tab[2]), 1095)])
            except:
                print ('data not imported: ', data)    
    
    for i in range(20):
        print (result[i])
    
    neural = NeuralNetwork()
    for index in range(200):
        for i in range(0, len(result)):
            neural.train(result[i][:2], result[i][2:])
        print ('epoc:', index, '    result: ', neural.query(creatdata('10/05/2017;09:20'))*1095)    
    print (creatdata('10/03/2017;09:20'))
    ret = neural.query(creatdata('10/05/2017;09:20'))
    print (ret*1095)

