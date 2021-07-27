"""
MIT License

Copyright (c) 2021 Luke LaBonte

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


import numpy as np
from copy import deepcopy
import pickle


#Class to hold all information about the network
class network:
    def __init__(self, structure, learningrate):
        self.weights = []
        self.biases = []
        self.values = []
        self.valueswithoutsigmoid = []
        self.structure = structure
        self.lr = learningrate
        
        #Set random weights and biases in the shape of the structure
        self.biases = [np.random.randn(y, 1) for y in structure[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(structure[:-1], structure[1:])]
    
    def train(self, netinput, output):
        error = []
        #Loop through all values in the input
        for i in range(len(netinput)):
            #Forward propogate values to get the result
            result = self.forwardprop(netinput[i])
            #Calculate error
            error = self.calcerror(result, output[i])
            #Backpropogate the network to change weights and biases
            self.backprop(error)
        return result, error

    def run(self, input):
        return self.forwardprop(input)
            

    def forwardprop(self, input):
        #values and valueswithoutsigmoid need to be reset here, or values will just be appended to the end
        self.values = []
        self.valueswithoutsigmoid = []
        #Loop through the layers in the network
        for i in range(len(self.structure)):
            #If we are on the first layer, just the input values should be appended. No matrix multiplicaiton.
            if(i == 0):
                self.values.append(sigmoid(input))
                self.valueswithoutsigmoid.append(input)
            
            #On any other layer, the values should be multiplied with the weights of the previous layer
            else:
                #Multiply weights with values
                dotp = np.dot(self.weights[i-1], self.values[-1])
                #Add bias
                dotp = elementwiseadd(dotp, self.biases[i-1])
                #Append new values to the value array
                self.valueswithoutsigmoid.append(dotp)
                self.values.append(sigmoid(dotp))
        return self.values[-1]

    def backprop(self, totalerror):
        #Deepcopy is necessary here so that editing one object will not change the other
        self.newweights = deepcopy(self.weights)
        self.newbiases = deepcopy(self.biases)
        error = []
        #Run through layers of network, starting with the last layer and working backwards
        for i in range(len(self.structure)-1):
            if i == 0:
                #If we are on the last layer, the error derivative is just the totalerror calculated before
                error = totalerror
            else:
                #If we are not on the last layer, the error derivative is calculated through a dot product of the transposed weight matrix of the previous layer and the current error
                wd = np.dot(np.transpose(self.weights[0 - i]), error)
                error = np.multiply(wd, sigmoidderiv(self.valueswithoutsigmoid[-1 - i]))
                
            #Loop through the neurons of the current layer
            for j in range(len(self.weights[-1 - i])):
                #Loop through weights of each neuroon in the current layer
                for k in range(len(self.weights[-1 - i][j])):
                    #Set the current weight to the partial derivative of the cost function
                    newweight = self.values[-2 - i][k] * error[j] * self.lr
                    self.newweights[-1 - i][j][k] -= newweight
            #Set biases equal to the current biases minus the error for that layer
            self.newbiases[-1 - i] = np.subtract(self.biases[-1 - i], np.reshape(error, (-1, 1))) * self.lr

            
        #Set the weights and biases equal to the new weights and biases calculated above
        self.weights = deepcopy(self.newweights)
        self.biases = deepcopy(self.newbiases)
            

    def calcerror(self, result, desiredresult):
        #Calculate the error between the result we got and the result we want
        return np.multiply((result - desiredresult), sigmoidderiv(self.valueswithoutsigmoid[-1]))


#Helper function to apply the sigmoid function to an array
def sigmoid(num):
    arr = deepcopy(num)
    for i in range(len(arr)):
        arr[i] = 1.0 / (1 + np.exp(-arr[i]))
    return arr

#Helper function to apply the sigmoid function to a single number
def singlesig(num):
    return 1.0 / (1 + np.exp(-num))


#Helper function to apply the derivative of the sigmoid function to an array
def sigmoidderiv(result):
    final = []
    for i in range(len(result)):
        final.append(singlesig(result[i]) * (1 - singlesig(result[i])))
    return final


#Helper function to add two list elementwise
def elementwiseadd(list1, list2):
    if len(list1) == 0:
        return list2
    else:
        for i in range(len(list1)):
            list1[i] += list2[i]
        return list1


def train(structure, input, output, learningrate, epochs):
    net = network(structure, learningrate)

    #Loop through each training example
    for j in range(epochs):
        networkout = net.train(input, output)
        #Print error and save the current network (You can take out the print statement, but I recommend keeping the pickle dump just in case)
        if j % 500 == 0 and j != 0:
            print('Error:', abs(networkout[1]), "Epoch", j, "out of", epochs)
            file = open('network.obj', 'wb+')
            pickle.dump(net, file, protocol=-1)
            file.close()

    #Dump trained network to pickle file
    file = open('network.obj', 'wb+')
    pickle.dump(net, file, protocol=-1)
    file.close()
    return net



#Exact same as previous function, except instead of starting from scratch, a partially trained network resumes training
def resume(network, input, output, learningrate, epochs):
    net = network

    for j in range(epochs):
        networkout = net.train(input, output)
        if j % 500 == 0 and j != 0:
            print('Error:', abs(networkout[1]), "Epoch", j, "out of", epochs)
            file = open('network.obj', 'wb+')
            pickle.dump(net, file, protocol=-1)
            file.close()

    file = open('network.obj', 'wb+')
    pickle.dump(net, file, protocol=-1)
    file.close()
    return net
