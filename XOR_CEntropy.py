#!/usr/bin/python

import numpy as np
from pyNNGraph import *

def get_progress_bar(loss, maxLoss, scale=50):
    n = int(scale*loss/maxLoss)
    s = '|'+'X'*n+(scale-n)*'-'+'|'
    return s

if __name__ == "__main__":
    """To solve the XOR problem, let's build the following net:

        Xin(2) --->Linear(2,3)--->Tanh(3)--->Linear(3,2)---> Out(2)

    """

    #Allocating all the nodes we need
    linear1 = Linear(2, 3)
    linear2 = Linear(3, 2)
    tanh = Tanh(3)

    #We will use CE as error metric
    err1 = CELayer(2)

    nodesTable = {'linear1':linear1, 'tanh':tanh, 'linear2':linear2}

    inputNodes={'linear1'}
    outputNodes={'linear2'}
    evaluationSequence=['tanh', 'linear2']

    myNet = Network(nodesTable, inputNodes, outputNodes, evaluationSequence)

    #Creating the network
    myNet.link_nodes('linear1', 'tanh')
    myNet.link_nodes('tanh', 'linear2')
    #myNet.link_nodes('linear2', 'tanh')

    #A XOR dataset
    data = [(0.,0.), (1.,0.), (1.,1.), (0.,1.)]
    classes = [0, 1, 0, 1]

    #get lists of references to parameters of the network
    params, gradParams = myNet.get_link_to_parameters()

    #Evaluation function: perform one epoch 
    def feval(x):

        myNet.reset_grad_param()

        errSum = 0.
        for cx,ct in zip(data, classes): 

            currentInput = np.array(cx)
            currentClass = ct

            outs = myNet.forward([currentInput]) #Push the input inside the network
            errSum += err1.forward(outs[0], currentClass)
            gradOutput1 = err1.backward(outs[0], currentClass)

            myNet.backward([currentInput], [gradOutput1])

        return errSum, gradParams


    #Check if the gradient is ok (the values should be approx. 1e-5)
    relativeErr = myNet.gradient_checking()
    print "\n\nChecking the gradient computation ... ",
    if sum([1 if error>5e-4 else 0 for error in relativeErr]) != 0:
        print "gradient wrong.", relativeErr
    else:
        print "gradient OK."

    print "\n\nTraining ...\n"

    maxLoss = 0.


    #Training routine:
    optimConf = {'learningRate':0.5, 'learningRateDecay':0.01, 'momentum':0.2, 'weightDecay':0.0}
    optimState = {} #Just a container to save the optimization related variables (e.g. the previous gradient...)

    for it in xrange(10000):
        loss = SGD(feval, params, optimConf, optimState)
        maxLoss = max(loss, maxLoss)
        print "epoch #"+str(it)+"\t"+get_progress_bar(loss, maxLoss)+' '+str(loss)
        if loss < 1e-1:
            print "\nTraining over."
            break

    #Displaying the input/output pairs:
    print "\nChecking the output for each input:"
    for cx,ct in zip(data, classes): 
        currentInput = np.array(cx)
        currentTarget = np.array(ct)
        outs = myNet.forward([currentInput])
        prob = err1.get_probabilities(outs[0])
        print "\tinput",cx,"yields output",prob

    print ""


    """ EXCPECTED OUTPUT:


    Checking the gradient computation ...  gradient OK.


    Training ...

    epoch #0    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX| 4.12227878586
    epoch #1    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----| 3.72330040968
    epoch #2    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-------------| 3.07850003418
    epoch #3    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------------------| 2.4284538207
    epoch #4    |XXXXXXXXXXXXXXXXXXXXXXXXXX------------------------| 2.19746653749
    epoch #5    |XXXXXXXXXXXXXXXXXXXXXXXXXX------------------------| 2.16294801591
    epoch #6    |XXXXXXXXXXXXXXXXXXXXXX----------------------------| 1.82532037197
    epoch #7    |XXXXXXXXXXXXXXXXXXXXXXX---------------------------| 1.9339024829
    epoch #8    |XXXXXXXXXXXXXXXXXXXX------------------------------| 1.72104115587
    epoch #9    |XXXXXXXXXXXXXXXXXXXXXXXX--------------------------| 1.99878457322
    epoch #10   |XXXXXXXXXXXXXXXXX---------------------------------| 1.47336125324
    epoch #11   |XXXXXXXXXXXXXXXXX---------------------------------| 1.45641687419
    epoch #12   |XXXXXXXXXX----------------------------------------| 0.867332497265
    epoch #13   |XXXXXXX-------------------------------------------| 0.601667374996
    epoch #14   |XXXXX---------------------------------------------| 0.476348221802
    epoch #15   |XXXXX---------------------------------------------| 0.430429206545
    epoch #16   |XXXX----------------------------------------------| 0.392991010062
    epoch #17   |XXXX----------------------------------------------| 0.361379128842
    epoch #18   |XXXX----------------------------------------------| 0.334339611754
    epoch #19   |XXX-----------------------------------------------| 0.310965672746
    epoch #20   |XXX-----------------------------------------------| 0.290577131399
    epoch #21   |XXX-----------------------------------------------| 0.272651567208
    epoch #22   |XXX-----------------------------------------------| 0.256780484858
    epoch #23   |XX------------------------------------------------| 0.242639761038
    epoch #24   |XX------------------------------------------------| 0.229968984418
    epoch #25   |XX------------------------------------------------| 0.218556629257
    epoch #26   |XX------------------------------------------------| 0.208229194394
    epoch #27   |XX------------------------------------------------| 0.198843110688
    epoch #28   |XX------------------------------------------------| 0.190278622941
    epoch #29   |XX------------------------------------------------| 0.182435105275
    epoch #30   |XX------------------------------------------------| 0.175227433159
    epoch #31   |XX------------------------------------------------| 0.168583144798
    epoch #32   |X-------------------------------------------------| 0.162440199294
    epoch #33   |X-------------------------------------------------| 0.156745190837
    epoch #34   |X-------------------------------------------------| 0.151451914837
    epoch #35   |X-------------------------------------------------| 0.146520208113
    epoch #36   |X-------------------------------------------------| 0.141915004297
    epoch #37   |X-------------------------------------------------| 0.137605559543
    epoch #38   |X-------------------------------------------------| 0.133564814005
    epoch #39   |X-------------------------------------------------| 0.129768862276
    epoch #40   |X-------------------------------------------------| 0.126196511838
    epoch #41   |X-------------------------------------------------| 0.122828913033
    epoch #42   |X-------------------------------------------------| 0.119649247485
    epoch #43   |X-------------------------------------------------| 0.116642464542
    epoch #44   |X-------------------------------------------------| 0.113795057385
    epoch #45   |X-------------------------------------------------| 0.111094872044
    epoch #46   |X-------------------------------------------------| 0.108530943862
    epoch #47   |X-------------------------------------------------| 0.106093356931
    epoch #48   |X-------------------------------------------------| 0.103773122849
    epoch #49   |X-------------------------------------------------| 0.10156207579
    epoch #50   |X-------------------------------------------------| 0.0994527813969

    Training over.

    Checking the output for each input:
        input (0.0, 0.0) yields output [ 0.9829254  0.0170746]
        input (1.0, 0.0) yields output [ 0.02772653  0.97227347]
        input (1.0, 1.0) yields output [ 0.98387857  0.01612143]
        input (0.0, 1.0) yields output [ 0.03521061  0.96478939]



    """