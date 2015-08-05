#PyNNGraph

Following the versatility of torch.nngraph, pyNNGraph allows you to create multiple inputs/outputs feed-forward and recurrent neural networks.

PyNNGraph is very similar to torch, in order to train a network you need four things: 

* A network i.e. a function with some parameters to tune. 
* An error metric to evaluate how good the network is performing on some input, knowing the expected output.
* An evaluation function to circle through the dataset and compute the gradient of the error with respect to the parameters of the networks. 
* An optimization method to minimize the error by tuning the parameters of the network accordingly. 

##Building a network

With pyNNGraph you can create complex networks by combining multiples modules together. For example here is a network that can be used to solve the XOR problem:

    in(2) --->Linear(2,2)--->Sigmoid(2)--->Linear(2,1)--->Tanh(1)---> out(1)

To build this network we can use the Network constructor, the required parameters are:

* nodesTable: a dictionary containing all the nodes we need associated with an alias. 

    ```python
    nodesTable = {
                  'firstLayer_lin': Linear(2,2), 
                  'firstLayer_squash': Sigmoid(2), 
                  'secondLayer_lin': Linear(2,1),
                  'secondLayer_squash': Tanh(1),
                  }
    ```

* inputNodes: a list containing the aliases of all the input nodes.

    ```python
    inputNodes = ['firstLayer_lin']
    ```
  
* outputNodes: a list containing the aliases of all the output nodes.

    ```python
    outputNodes = ['secondLayer_squash']
    ```
  
* evaluationSequence: a list of aliases showing the order in which we should evaluate the nodes in order to push some input forward in the network. The inputs nodes are not included in the evaluation sequence. 

    ```python
    evaluationSequence = ['firstLayer_squash', 'secondLayer_lin', 'secondLayer_squash']
    ```

We can call initialize the network using these elements `net = Network(nodesTable, inputNodes, outputNodes, evaluationSequence)`. We still need to link the nodes together:

```python
myNet.link_nodes('firstLayer_lin', 'firstLayer_squash')
myNet.link_nodes('firstLayer_squash', 'secondLayer_lin')
myNet.link_nodes('secondLayer_lin', 'secondLayer_squash')
```

The whole point of having this repetitive synthax is to have versatility in the networks we can create. A network can have several inputs/outputs, cross or split hidden states as you wish. In the future it should be possible to automatically compute the evaluation sequence. The description of each module should be available soon in the module folder. 

Once the network has been initialized, we can use the `forward` and `backward` methods. `forward(inputList)` computes the outputs of the network for the given `inputList` of inputs. The order inside this list should match the `inputNodes` list previously defined. `backward(inputList, gradOutputs)` propagates the error derivatives backward in the network, it is responsible for __accumulating__ the gradient of the error with respect to the parameters of the network. Its parameters are the `inputList` as previously introduced, and `gradOutputs`, the derivative or the error with respect to the outputs of the network. 

##Getting the error derivatives

PyNNGraph only have two error metrics available at the moment:

* Mean Squared Error: the good old MSE, `MSE(netOutput, targetVec) = (netOutput - targetVec)^2`

    ```python
    MSE = MSELayer(outputSize)
    errorVec = MSE.forward(netOutput, targetVec) 
    dEdOuts = MSE.backward(netOutput, targetVec) #compute the gradient wrt. the output of the net
    ```

* Cross-entropy Error: usefull for classification tasks,
  `CEE(netOutput, tClass) = -log(softmax(netOutput)[tClass])`

    ```python
    CEE = CELayer(outputSize)
    errorVec = MSE.forward(netOutput, targetClass) #targetClass is an integer
    dEdOuts = MSE.backward(netOutput, targetClass) 
    ```

##Building an evaluation function

The role of the evaluation function is to compute the error and gradient for one minibatch. The prototype of this function should be `loss, gradParams = feval(params)` where `loss` is the error return for the current minibatch, `gradParams` and `params` are lists of references to respectively the parameters' gradient and the parameters (can be obtained with `Network.get_link_to_parameters()`). `params` represent the state of a network, it is `params` we want to tune. 

An example of evaluation function:

```python

params, gradParams = myNet.get_link_to_parameters()
  
def feval(x):
    #eventually modify params if params != x
    myNet.reset_grad_param() #remember the gradient is accumulated
    err = 0.
    for cx,ct in zip(inputs, targets): #circling over the entire dataset 
        outs = myNet.forward(cx)    #Push the input inside the network
        err += np.mean(MSE.forward(outs[0], ct))
        gradOutput = MSE.backward(outs[0], ct) #gradOutput = dE/douts
        myNet.backward([cx], [gradOutput])
    err /= len(inputs)
    return err, gradParams
    
```
The evaluation function is given to the optimizer.

##Choosing an optimizer

The role of the optimizer is to tune the parameters of the network to minimize the error metric. The most classic method to do thid is gradient descent. A more performant method for classification tasks might be RMSprop. Both methods are implemented in pyNNGraph. Each optimizer has the following prototype:

```python
loss = SGD(feval, netParams, optimConf, optimState)
```
Where `optimConf` is a hash table containing the parameters of the optimization method, and `optimState` is a hash table container used by the optimizer to keep track of some parameters such as the previous gradient for `SGD`, used to implement momentum. 

```python
optimConf = {'learningRate':0.5, 'learningRateDecay':0.01, 'momentum':0.9, 'weightDecay':0.0}
optimState = {}

for it in xrange(1000): #perform the training
    loss = SGD(feval, params, optimConf, optimState)
```

##Recurrent Neural Networks

The real goodness of pyNNGraph starts here, it is really easy to create and train recurrent networks. Simply create a recurrent connexion where needed:

```python
net.recurrent_connexion('nodeAlias1', 'nodeAlias2')
```

To train the network you can then unwrap the network and handle it as you would handle a feed-forward net. For example, to build and train a simple RNN with one hidden layer:

```python
nodesTable = {
              'frwdLinear': Linear(4, 3),
              'recLinear': Linear(3, 3),
              'cellSquash': Tanh(3),
              'add': CAddTable(3),
              'outLinear': Linear(3, 4),
              'outSquash': Tanh(4)
            }
inputNodes = ['frwdLinear']
outputNodes = ['outSquash']
evaluationSequence = ['recLinear', 'add', 'cellSquash', 'outLinear', 'outSquash']

net = Network(nodesTable, inputNodes, outputNodes, evaluationSequence)

net.link_nodes('frwdLinear', 'add')
net.link_nodes('recLinear', 'add')
net.link_nodes('add', 'cellSquash')
net.link_nodes('cellSquash', 'outLinear')
net.link_nodes('outLinear', 'outSquash')

net.recurrent_connexion('cellSquash', 'recLinear')

#get lists of references to parameters of the network
params, gradParams = myNet.get_link_to_parameters()

#Choosing an error metric
CEE = CELayer(4)

myNet.unwrap(4) #unwrap the network on 4 timesteps

def feval(x):
    myNet.reset_grad_param()
    err = 0.

    seq, targetSeq = get_next_sample()

    #FORWARD:
    outs = myNet.forward(seq)
    err = sum([CEE.forward(outs[i], targetSeq[i]) for i in xrange(4)])/4.0

    #BACKWARD:
    gradOutputs = [CEErr.backward(outs[i], targetSeq[i]) for i in xrange(4)]
    myNet.backward(seq, gradOutputs)
    
    return err, gradParams
    
optimConf = {'learningRate':0.1, 'decayRate':0.98}
optimState = {}

for it in xrange(300):
    loss = RMSprop(feval, params, optimConf, optimState)
```

Once the training is done, the `Network.sequence_prediction(inputList)` method can be used to continously feed data to the network. `Network.reset_recurrent_states()` resets the recurrent states to zeros. 

Check test_rnn.py and test_lstm.py for more examples. 

##Building a language model

Work in progress 
