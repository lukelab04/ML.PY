# About ML.PY
ML.PY is a Python library for machine learning. Though not as powerful as libraries like TensorFlow, there is almost nothing simpler or eaiser to use than this. Taking out blank lines, the example in this file trains and tests a model in just 25 lines of code.
## Depedancies
Numpy (Linear algebra), Pickle (Object storage)

# Usage
Here, I will run through the process of creating a network that learns an XOR gate.

The first thing we will do is import our dependancies. You will need to download the network.py file into your project directory.
```
import pickle
import network
from copy import deepcopy
```

Next, we will define the input, output, and structure of the network. The structure defines both the number of layers and the amount of neurons in those layers. This network has 2 inputs, one hidden layer with 6 neurons, and 1 output. The number of inputs and outputs corresponds to the number of values in each subarray of the netinput and netoutput variable.
```
netinput = [[0, 0],[1, 0],[0, 1],[1, 1]]
netoutput = [[0],[1],[1],[0]]
structure = [2, 6, 1]
```


The next step is training the model. There are two functions we will use: train and resume. Resume takes a partially trained network as a paremeter, whereas train does not.

With such a simple network, we will not need very many training iterations.

```
for i in range(3000):
  print(i)
  if(i == 0):
    net = deepcopy(network.train(structure, netinput, netoutput, 0.1, 10))
  else:
    net = deepcopy(network.resume(net, netinput, netoutput, 0.1, 10))
```
The last argument in both functions defines how many times the network should loop over individual training points. The for loop range (3000 in this example) defines how many total times the network will run through the entire dataset.

Next, we will load the trained network from the pickle file. The pickle file is always stored as 'network.obj' in the same directory as the network.py file.

```
file = open('network.obj', 'rb') //rb is necessary here because the file is in binary format
net = pickle.load(file)
file.close()
```

Finally, we will test the accuracy of the network. This is purely for the sake of demonstrating the final product. In real life, you would probably write something to implement the network instead.

```
right = 0
wrong = 0

for i in range(len(netinput)):
  out = net.run(netinput[i])
  if(round(out[0]) == netoutput[i][0]):
    right += 1
   else:
    wrong += 1
print("right:", right, "wrong:", wrong)
```

If everything goes right, this network should correctly pick the output of an XOR gate when given the input. This is a very simple example, and more complex networks with several hidden layers can be created for more complicated datasets. Here is the final code:

```
import pickle
import network
from copy import deepcopy


netinput = [[0, 0],[1, 0],[0, 1],[1, 1]]
netoutput = [[0],[1],[1],[0]]
structure = [2, 6, 1]

for i in range(3000):
    print(i)
    if(i == 0):
        net = deepcopy(network.train(structure, netinput, netoutput, 0.1, 10))
    else: 
        net = deepcopy(network.resume(net, netinput, netoutput, 0.1, 10))

file = open('network.obj', 'rb')
net = pickle.load(file)
file.close()

right = 0
wrong = 0

for i in range(len(netinput)):
    out = net.run(netinput[i])
    if(round(out[0]) == netoutput[i][0]):
        right+=1
    else:
        wrong += 1
print("right: ", right, "wrong: ", wrong)
```
