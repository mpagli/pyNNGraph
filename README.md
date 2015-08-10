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

Inspired from [char-rnn](https://github.com/karpathy/char-rnn) we can build a LSTM character-based language model. The dataset we are going to use consist of most of the Lovecraft novels. This is a rather small 1.1Mo dataset, here is a sample:

> The older matters which had made the sculptor's dream and bas-relief so significant to my uncle formed the subject of the second half of his long manuscript. Once before, it appears, Professor Angell had seen the hellish outlines of the nameless monstrosity, puzzled over the unknown hieroglyphics, and heard the ominous syllables which can be rendered only as "Cthulhu"; and all this in so stirring and horrible a connexion that it is small wonder he pursued young Wilcox with queries and demands for data.

The vocabulary is very rich and the sentences often quite long. Let's see how far we can get from this. We are going to use a two layer LSTM network, the size of the hidden layers is 200. Here are some results for different epochs:

Initially, the network outputs a random character distribution:

> krn30cnFvnggY1OZ7Kw?P8-5(C4-f1yplDc8.F2994,ifjhvWevDGwJ wE7ciQVp!Fsir2 pH"sPXUZs8vi)MNaSCd,yCFmSGC1,N,ni'FBv1:Eo'nsOsKEjlIisOF6Zi:o0:!QGETSa(C wZN'BGvS(yHAu( rFo!wtTyrulKz1wfimwujFy4-O4p)lsL17IA qq!fF

After 200 epochs, the network has learnt a correct distribution:

> ere an ar on de ee fhe bn or fn th the geit an the th bo aa tiim on won won wh we bas and tie on on on n tone an vas an th we ar bh w ta tas the whe whe we th we fore se ton wh ch te an th the toce wh

After 400 epochs, some words start to make sense:

> on anian he of of anpe sas an on the on the and the the seng an pofend he the Sone he fone I the of the on and and on ar the and the the on he af and on the whe the fed an and Ceror had fed and and wh

After 3000 epochs:

> ricederinger. The the and whin Was pring the was the frous the - be rein I he se the was in the valed of the hadestered be a coul congur he had the pren whin wher of andared dedered be fren a berered

After 12400 epochs: 

> an very and know of the shere along which and suld from by the sees the mose with and young he had the starich all the of the of the screst sunde and down the clus Allen I was the expenter - and by th

Finally here is some text obtained from sampling the trained network:

> strent up the with me he had a concert was a surved sugher the man and which he great and had found dement the had be stent the centre were he sughed the could the fling and the sten and lay and the special the skes of the great and the which the and and the of a the spin the rest sughing I was a distry of the for of the couring in the spunder my frouly be the before the dest which the sugher explate of the were a whin a southere the had the part the but he was the and muse and the regular and with and all the the lest a speched the strang the clanged of the speculated the wandering by the for for the perivered and and the time the speched the carese stoned the cres and the could the maning

If it looks and sounds like english, the meaning is missing. Adding more parameters to our network might help. With a hidden size of 300 (previously 200):

> he distrestering in a langed and in whiched by a comples the proved the specisive more resellen the indistanced the contround the but whiching the selinis were present thought such and the gedered the care. The light could he surfous mounted the of the distresting from and his his frour and all in the chered the were beside of a cloud the dreaded the but the still the sound such distingulfor began the noring of a distrestering of the he of the sound and decighteraller in the cry his late minoread and and fear of the mongrend the greater of one whichesher of the best became of the strend whiches the and and a bear the began's specimen the of the lest which he was the strend and with the such 

With a hidden size of 400:

> nd spect and of his planed the present my see of it which and in which in expecting of the for his under with the such a speciverest of a with the better while in the bear and special southare the strent and and was not and vere of which the long the regularity which the could could not and a some intent of the possin which we had a while and and will of a secured the more and a had to a comples the hands and seen the more with the providere and from a wing the country were the bestrand in the wing new and the and sent drears of the covered the mountains and found of the subtly and and the share of the seare the men to a sea from a many descrip of a light the supply of the of the are be the 

Adding parameters allowed the rnn to learn more complicated features. The vocabulary is richer and the validation error went lower. For a small and complex dataset like ours, these results are satisfying. 





