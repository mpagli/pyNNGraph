#!/usr/bin/python

import numpy as np
from pyNNGraph import *

if __name__ == "__main__":
    """Let's build the following network:

                                 |---Linear(3, 1)---Tanh(1)--- Y1 -> MSE
 Xin ---Linear(2, 3)---Tanh(3)---|
                                 |---Linear(3, 1)---Tanh(1)--- Y2 -> MSE

    """

    #Allocating all the nodes we need
    linear1 = Linear(2, 3)
    tanh1 = Tanh(3)
    linear2 = Linear(3, 1)
    linear3 = Linear(3, 1)
    tanh2 = Tanh(1)
    tanh3 = Tanh(1)
    err1 = MSELayer(1)
    err2 = MSELayer(1)

    nodesTable = {'linear1':linear1, 'tanh1':tanh1, 'linear2':linear2, 'tanh2':tanh2, 'linear3':linear3, 'tanh3':tanh3}

    myNet = Network(nodesTable, {'linear1'}, {'tanh2','tanh3'}, ['tanh1' ,'linear2', 'tanh2', 'linear3', 'tanh3'])

    #Creating the network
    myNet.link_nodes('linear1', 'tanh1')
    myNet.link_nodes('tanh1', 'linear2')
    myNet.link_nodes('linear2', 'tanh2')
    myNet.link_nodes('tanh1', 'linear3')
    myNet.link_nodes('linear3', 'tanh3')

    #A very small dataset
    data = [(0.,0.), (1.,0.), (1.,1.)]
    t    = [(1.),    (1.),    (-1.)]

    #get lists of references to parameters of the network
    params, gradParams = myNet.get_link_to_parameters()

    #Evaluation function: perform one epoch 
    def feval(x):

        myNet.reset_grad_param()

        errSum = 0.
        for cx,ct in zip(data, t): 

            currentInput = np.array(cx)
            currentTarget = np.array(ct)

            outs = myNet.forward([currentInput]) #Push the input inside the network

            errSum += err1.forward(outs[0], currentTarget)
            errSum += err2.forward(outs[1], currentTarget)

            gradOutput1 = err1.backward(outs[0], currentTarget)
            gradOutput2 = err2.backward(outs[1], currentTarget)

            myNet.backward([currentInput], [gradOutput1,gradOutput2])

        return errSum, gradParams

    optimConf = {'learningRate':0.1, 'learningRateDecay':0.01, 'momentum':0.0, 'weightDecay':0.0}
    optimState = {}

    #Check if the gradient is ok (the values should be approx. 1e-5)
    relativeErr = myNet.gradient_checking()
    print "\n\nChecking the gradient computation ... ",
    if sum([1 if error>1e-4 else 0 for error in relativeErr]) != 0:
        print "gradient wrong.", 
    else:
        print "gradient OK."

    print "\n\nTraining ...\n"
    
    #Training routine (50 epochs here)
    for it in xrange(50):
        loss = SGD(feval, params, optimConf, optimState)
        print "epoch ",it," loss: ",loss

