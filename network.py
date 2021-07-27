import numpy as np
from copy import deepcopy
import pickle



class network:
    def __init__(self, structure, learningrate):
        self.weights = []
        self.biases = []
        self.values = []
        self.valueswithoutsigmoid = []
        self.structure = structure
        self.lr = learningrate
        
        self.biases = [np.random.randn(y, 1) for y in structure[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(structure[:-1], structure[1:])]
    
    def train(self, netinput, output):
        error = []
        for i in range(len(netinput)):
            result = self.forwardprop(netinput[i])
            # error = elementwiseadd(error, self.calcerror(result, output[i]))
            error = self.calcerror(result, output[i])
            self.backprop(error)
        return result, error

    def run(self, input):
        return self.forwardprop(input)
            

    def forwardprop(self, input):
        self.values = []
        self.valueswithoutsigmoid = []
        for i in range(len(self.structure)):
            if(i == 0):
                self.values.append(sigmoid(input))
                self.valueswithoutsigmoid.append(input)
            else:
                dotp = np.dot(self.weights[i-1], self.values[-1])
                dotp = elementwiseadd(dotp, self.biases[i-1])
                self.valueswithoutsigmoid.append(dotp)
                self.values.append(sigmoid(dotp))
        return self.values[-1]

    def backprop(self, totalerror):
        self.newweights = deepcopy(self.weights)
        self.newbiases = deepcopy(self.biases)
        error = []
        for i in range(len(self.structure)-1):
            if i == 0:
                error = totalerror
            else:
                wd = np.dot(np.transpose(self.weights[0 - i]), error)
                error = np.multiply(wd, sigmoidderiv(self.valueswithoutsigmoid[-1 - i]))
            for j in range(len(self.weights[-1 - i])):
                for k in range(len(self.weights[-1 - i][j])):
                    newweight = self.values[-2 - i][k] * error[j] * self.lr
                    self.newweights[-1 - i][j][k] -= newweight
            self.newbiases[-1 - i] = np.subtract(self.biases[-1 - i], np.reshape(error, (-1, 1))) * self.lr

        self.weights = deepcopy(self.newweights)
        self.biases = deepcopy(self.newbiases)
            

    def calcerror(self, result, desiredresult):
        return np.multiply((result - desiredresult), sigmoidderiv(self.valueswithoutsigmoid[-1]))


def sigmoid(num):
    arr = deepcopy(num)
    for i in range(len(arr)):
        arr[i] = 1.0 / (1 + np.exp(-arr[i]))
    return arr
def singlesig(num):
    return 1.0 / (1 + np.exp(-num))

def sigmoidderiv(result):
    final = []
    for i in range(len(result)):
        final.append(singlesig(result[i]) * (1 - singlesig(result[i])))
    return final

    
def elementwiseadd(list1, list2):
    if len(list1) == 0:
        return list2
    else:
        for i in range(len(list1)):
            list1[i] += list2[i]
        return list1



def train(structure, input, output, learningrate, epochs):
    net = network(structure, learningrate)

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
