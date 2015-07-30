#!/usr/bin/python

import numpy as np
from pyNNGraph import *

def get_progress_bar(loss, maxLoss, scale=50):
    n = int(scale*loss/maxLoss)
    s = '|'+'X'*n+(scale-n)*'-'+'|'
    return s

if __name__ == "__main__":
    """"""

    #Allocating all the nodes we need
    #------- For the rnn cell
    frwdLinear = Linear(4, 4)
    recLinear = Linear(4, 4)
    recSquash = Tanh(4)
    add = CAddTable(4)
    #------- For the output layer
    outLinear = Linear(4,4)
    CEErr = CELayer(4)

    nodesTable = {'frwdLinear':frwdLinear, 'recLinear':recLinear, 'recSquash':recSquash, 'add':add, 'outLinear':outLinear}

    inputNodes = ['frwdLinear']
    outputNodes = ['outLinear']
    evaluationSequence = ['recLinear', 'add', 'recSquash', 'outLinear']

    myNet = Network(nodesTable, inputNodes, outputNodes, evaluationSequence)

    #Creating the network
    myNet.link_nodes('frwdLinear', 'add')
    myNet.link_nodes('recLinear', 'add')
    myNet.link_nodes('add', 'recSquash')
    myNet.link_nodes('recSquash', 'outLinear')

    myNet.recurrent_connexion('recSquash', 'recLinear') #create a recurrent connexion between the output of 
                                                        #recSquash and the input of recLinear

    #get lists of references to parameters of the network
    params, gradParams = myNet.get_link_to_parameters()

    myNet.unwrap(4) #unwrap the network on 4 timesteps 

    print "\nNetwork unwrapped, \n\tinputs:",myNet.inputNodes,"\n\toutputs",myNet.outputNodes

    #A small dataset, the word 'hello': 
    seq = [[1.,0.,0.,0.], [0.,1.,0.,0.], [0.,0.,1.,0.], [0.,0.,1.,0.], [0.,0.,0.,1.]]
    seq = [np.array(x) for x in seq]
    classes = [1, 2, 2, 3, 0]

    #Evaluation function: perform one epoch 
    def feval(x):

        myNet.reset_grad_param()
        errSum = 0.

        #FORWARD:
        outs = myNet.forward(seq)

        errSum = sum([CEErr.forward(outs[i], classes[i]) for i in xrange(4)])

        #BACKWARD:
        gradOutputs = [CEErr.backward(outs[i], classes[i]) for i in xrange(4)]
        myNet.backward(seq, gradOutputs)

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

    for it in xrange(100):
        loss = SGD(feval, params, optimConf, optimState)
        maxLoss = max(loss, maxLoss)
        print "epoch #"+str(it)+"\t"+get_progress_bar(loss, maxLoss)+' '+str(loss)
        if loss < 1e-1:
            print "\nTraining over."
            break

    #Displaying the input/output pairs:
    print "\nChecking the output for each input:"


    print ""


    """ EXCPECTED OUTPUT:

    Network unwrapped, 
        inputs: ['frwdLinear', 'frwdLinear_t1', 'frwdLinear_t2', 'frwdLinear_t3'] 
        outputs ['outLinear', 'outLinear_t1', 'outLinear_t2', 'outLinear_t3']


    Checking the gradient computation ...  gradient OK.


    Training ...

    epoch #0    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX| 7.60194759181
    epoch #1    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------| 6.18205639628
    epoch #2    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX| 7.99710825593
    epoch #3    |XXXXXXXXXXXXXXXXXXXXXXXXXX------------------------| 4.18008484613
    epoch #4    |XXXXXXXXXXXXXXXXXXXXXX----------------------------| 3.56194180528
    epoch #5    |XXXXXXXXXXXXXX------------------------------------| 2.2700264083
    epoch #6    |XXXXXXXXX-----------------------------------------| 1.58176753351
    epoch #7    |XXXXX---------------------------------------------| 0.851854656748
    epoch #8    |XXX-----------------------------------------------| 0.490873127585
    epoch #9    |XX------------------------------------------------| 0.346833745193
    epoch #10   |X-------------------------------------------------| 0.278418340003
    epoch #11   |X-------------------------------------------------| 0.237861285922
    epoch #12   |X-------------------------------------------------| 0.208631772567
    epoch #13   |X-------------------------------------------------| 0.186333510567
    epoch #14   |X-------------------------------------------------| 0.168668491071
    epoch #15   |--------------------------------------------------| 0.15428422634
    epoch #16   |--------------------------------------------------| 0.142321258355
    epoch #17   |--------------------------------------------------| 0.132202612564
    epoch #18   |--------------------------------------------------| 0.123524463528
    epoch #19   |--------------------------------------------------| 0.115994699231
    epoch #20   |--------------------------------------------------| 0.109396262678
    epoch #21   |--------------------------------------------------| 0.103564212727
    epoch #22   |--------------------------------------------------| 0.0983708030059

    Training over.

    Checking the output for each input:




    """