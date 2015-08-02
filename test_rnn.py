#!/usr/bin/python

import numpy as np
from pyNNGraph import *

def get_progress_bar(loss, maxLoss, scale=50):
    n = int(scale*loss/maxLoss)
    s = '|'+'X'*n+(scale-n)*'-'+'|'
    return s

def clip_gradient(gradParams, maxValue=5.):
    """"""
    for gradParam in gradParams:
        gradParam.clip(-maxValue,maxValue,gradParam)


if __name__ == "__main__":
    """In this example we are going to learn the word 'hello' using a simple rnn:

        
            letterIn ---> Linear(4,3) ---> CAddTable(3) ---> Tanh(3) ---> Linear(3,4) ---> Tanh ---> Cross-Entropy error
                                               ^                      |
                                               |                      |
                                               ----- Linear(3,3) ------                               
                                                  recurrent connexion
    """

    #Allocating all the nodes we need
    #------- For the rnn cell
    frwdLinear = Linear(4, 3)
    recLinear = Linear(3, 3)
    recSquash = Tanh(3)
    add = CAddTable(3)
    #------- For the output layer
    outLinear = Linear(3, 4)
    outTanh = Tanh(4)
    CEErr = CELayer(4)

    nodesTable = {'frwdLinear':frwdLinear, 'recLinear':recLinear, 'recSquash':recSquash, 'add':add, 'outLinear':outLinear, 'outTanh':outTanh}

    inputNodes = ['frwdLinear']
    outputNodes = ['outTanh']
    evaluationSequence = ['recLinear', 'add', 'recSquash', 'outLinear', 'outTanh']

    myNet = Network(nodesTable, inputNodes, outputNodes, evaluationSequence)

    #Creating the network
    myNet.link_nodes('frwdLinear', 'add')
    myNet.link_nodes('recLinear', 'add')
    myNet.link_nodes('add', 'recSquash')
    myNet.link_nodes('recSquash', 'outLinear')
    myNet.link_nodes('outLinear', 'outTanh')

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

        errSum = sum([CEErr.forward(outs[i], classes[i]) for i in xrange(4)])/4.0

        #BACKWARD:
        gradOutputs = [CEErr.backward(outs[i], classes[i]) for i in xrange(4)]
        myNet.backward(seq, gradOutputs)

        clip_gradient(gradParams)

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
    optimConf = {'learningRate':0.1, 'learningRateDecay':0.001, 'momentum':0.5, 'weightDecay':0.99}
    optimState = {} #Just a container to save the optimization related variables (e.g. the previous gradient...)

    for it in xrange(300):
        loss = SGD(feval, params, optimConf, optimState)
        maxLoss = max(loss, maxLoss)
        print "epoch #"+str(it)+"\t"+get_progress_bar(loss, maxLoss)+' '+str(loss)
        if loss < 1e-1:
            print "\nTraining over."
            break

    #Displaying the input/output pairs:
    dictionary = ['h','e','l','o']
    print "\nFeeding a sequence to the network:"
    print "\tFeeding the letter 'h', prediction:",
    outs = myNet.sequence_prediction([seq[0]])
    print dictionary[np.argmax(CEErr.get_probabilities(outs[0]))],CEErr.get_probabilities(outs[0])
    print "\tFeeding the letter 'e', prediction:",
    outs = myNet.sequence_prediction([seq[1]])
    print dictionary[np.argmax(CEErr.get_probabilities(outs[0]))],CEErr.get_probabilities(outs[0])
    print "\tFeeding the letter 'l', prediction:",
    outs = myNet.sequence_prediction([seq[2]])
    print dictionary[np.argmax(CEErr.get_probabilities(outs[0]))],CEErr.get_probabilities(outs[0])
    print "\tFeeding the letter 'l', prediction:",
    outs = myNet.sequence_prediction([seq[2]])
    print dictionary[np.argmax(CEErr.get_probabilities(outs[0]))],CEErr.get_probabilities(outs[0])

    print ""


    """ EXCPECTED OUTPUT:

    Network unwrapped, 
        inputs: ['frwdLinear', 'frwdLinear_t1', 'frwdLinear_t2', 'frwdLinear_t3'] 
        outputs ['outTanh', 'outTanh_t1', 'outTanh_t2', 'outTanh_t3']


    Checking the gradient computation ...  gradient OK.


    Training ...

    epoch #0    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX| 6.83000725989
    epoch #1    |XXXXXXXXXXXXXXXXXXXXXXXXX-------------------------| 3.44796375203
    epoch #2    |XXXXXXXXXXXXXXXXXXXXXXXXX-------------------------| 3.42608458546
    epoch #3    |XXXXXXXXXXXXXXXXXXXXXXX---------------------------| 3.15882346371
    epoch #4    |XXXXXXXXXXXXXXXXXXXXXXXX--------------------------| 3.37534527184
    epoch #5    |XXXXXXXXXXXXXXXXXXXX------------------------------| 2.85462410392
    epoch #6    |XXXXXXXXXXXXXXXXXXX-------------------------------| 2.613265418
    epoch #7    |XXXXXXXXXXXXXXXXX---------------------------------| 2.4533857899
    epoch #8    |XXXXXXXXXXXXXXXXX---------------------------------| 2.37474513428
    epoch #9    |XXXXXXXXXXXXXXXXX---------------------------------| 2.36437775579
    epoch #10   |XXXXXXXXXXXXXXXXX---------------------------------| 2.35923666815
    epoch #11   |XXXXXXXXXXXXXXXXX---------------------------------| 2.35595066645
    epoch #12   |XXXXXXXXXXXXXXXXX---------------------------------| 2.35358399832
    epoch #13   |XXXXXXXXXXXXXXXXX---------------------------------| 2.35172052868
    epoch #14   |XXXXXXXXXXXXXXXXX---------------------------------| 2.35011054943
    epoch #15   |XXXXXXXXXXXXXXXXX---------------------------------| 2.34853977667
    epoch #16   |XXXXXXXXXXXXXXXXX---------------------------------| 2.34671361447
    epoch #17   |XXXXXXXXXXXXXXXXX---------------------------------| 2.34398238881
    epoch #18   |XXXXXXXXXXXXXXXXX---------------------------------| 2.33813677278
    epoch #19   |XXXXXXXXXXXXXXXX----------------------------------| 2.31581262872
    epoch #20   |XXXXXXXXXXXXXXX-----------------------------------| 2.10911024459
    epoch #21   |XXXXXXXXXXXXXX------------------------------------| 1.94319387373
    epoch #22   |XXXXXXXXXXXXX-------------------------------------| 1.89258909872
    epoch #23   |XXXXXXXXXXXXX-------------------------------------| 1.88809816993
    epoch #24   |XXXXXXXXXXXXX-------------------------------------| 1.8853586283
    epoch #25   |XXXXXXXXXXXXX-------------------------------------| 1.88338141438
    epoch #26   |XXXXXXXXXXXXX-------------------------------------| 1.88188418967
    epoch #27   |XXXXXXXXXXXXX-------------------------------------| 1.88072210654
    epoch #28   |XXXXXXXXXXXXX-------------------------------------| 1.87980641218
    epoch #29   |XXXXXXXXXXXXX-------------------------------------| 1.8790783508
    epoch #30   |XXXXXXXXXXXXX-------------------------------------| 1.87849710059
    epoch #31   |XXXXXXXXXXXXX-------------------------------------| 1.87803321604
    epoch #32   |XXXXXXXXXXXXX-------------------------------------| 1.87766476434
    epoch #33   |XXXXXXXXXXXXX-------------------------------------| 1.87737494233
    epoch #34   |XXXXXXXXXXXXX-------------------------------------| 1.87715056283
    epoch #35   |XXXXXXXXXXXXX-------------------------------------| 1.87698107014
    epoch #36   |XXXXXXXXXXXXX-------------------------------------| 1.87685788334
    epoch #37   |XXXXXXXXXXXXX-------------------------------------| 1.87677394402
    epoch #38   |XXXXXXXXXXXXX-------------------------------------| 1.87672339238
    epoch #39   |XXXXXXXXXXXXX-------------------------------------| 1.87670132552
    epoch #40   |XXXXXXXXXXXXX-------------------------------------| 1.87670361003
    epoch #41   |XXXXXXXXXXXXX-------------------------------------| 1.87672673276
    epoch #42   |XXXXXXXXXXXXX-------------------------------------| 1.87676767986
    epoch #43   |XXXXXXXXXXXXX-------------------------------------| 1.87682383782
    epoch #44   |XXXXXXXXXXXXX-------------------------------------| 1.87689291243
    epoch #45   |XXXXXXXXXXXXX-------------------------------------| 1.87697286203
    epoch #46   |XXXXXXXXXXXXX-------------------------------------| 1.87706184247
    epoch #47   |XXXXXXXXXXXXX-------------------------------------| 1.87715816122
    epoch #48   |XXXXXXXXXXXXX-------------------------------------| 1.87726023842
    epoch #49   |XXXXXXXXXXXXX-------------------------------------| 1.87736657294
    epoch #50   |XXXXXXXXXXXXX-------------------------------------| 1.87747571166
    epoch #51   |XXXXXXXXXXXXX-------------------------------------| 1.87758622005
    epoch #52   |XXXXXXXXXXXXX-------------------------------------| 1.87769665253
    epoch #53   |XXXXXXXXXXXXX-------------------------------------| 1.87780552041
    epoch #54   |XXXXXXXXXXXXX-------------------------------------| 1.87791125537
    epoch #55   |XXXXXXXXXXXXX-------------------------------------| 1.87801216541
    epoch #56   |XXXXXXXXXXXXX-------------------------------------| 1.87810637937
    epoch #57   |XXXXXXXXXXXXX-------------------------------------| 1.87819177456
    epoch #58   |XXXXXXXXXXXXX-------------------------------------| 1.87826587924
    epoch #59   |XXXXXXXXXXXXX-------------------------------------| 1.87832573773
    epoch #60   |XXXXXXXXXXXXX-------------------------------------| 1.87836771864
    epoch #61   |XXXXXXXXXXXXX-------------------------------------| 1.87838723541
    epoch #62   |XXXXXXXXXXXXX-------------------------------------| 1.87837832728
    epoch #63   |XXXXXXXXXXXXX-------------------------------------| 1.87833301223
    epoch #64   |XXXXXXXXXXXXX-------------------------------------| 1.87824025318
    epoch #65   |XXXXXXXXXXXXX-------------------------------------| 1.87808424127
    epoch #66   |XXXXXXXXXXXXX-------------------------------------| 1.87784141397
    epoch #67   |XXXXXXXXXXXXX-------------------------------------| 1.8774749912
    epoch #68   |XXXXXXXXXXXXX-------------------------------------| 1.87692429962
    epoch #69   |XXXXXXXXXXXXX-------------------------------------| 1.87608219944
    epoch #70   |XXXXXXXXXXXXX-------------------------------------| 1.87474237356
    epoch #71   |XXXXXXXXXXXXX-------------------------------------| 1.872459376
    epoch #72   |XXXXXXXXXXXXX-------------------------------------| 1.8681072881
    epoch #73   |XXXXXXXXXXXXX-------------------------------------| 1.85811082438
    epoch #74   |XXXXXXXXXXXXX-------------------------------------| 1.82651934428
    epoch #75   |XXXXXXXXXXXX--------------------------------------| 1.67638127422
    epoch #76   |XXXXXXXXXX----------------------------------------| 1.49188100037
    epoch #77   |XXXXXXXXXX----------------------------------------| 1.44943383232
    epoch #78   |XXXXXXXXXX----------------------------------------| 1.437588009
    epoch #79   |XXXXXXXXXX----------------------------------------| 1.43099172508
    epoch #80   |XXXXXXXXXX----------------------------------------| 1.42638319497
    epoch #81   |XXXXXXXXXX----------------------------------------| 1.42299122353
    epoch #82   |XXXXXXXXXX----------------------------------------| 1.42040906636
    epoch #83   |XXXXXXXXXX----------------------------------------| 1.41839476573
    epoch #84   |XXXXXXXXXX----------------------------------------| 1.41679445095
    epoch #85   |XXXXXXXXXX----------------------------------------| 1.41550529882
    epoch #86   |XXXXXXXXXX----------------------------------------| 1.41445584562
    epoch #87   |XXXXXXXXXX----------------------------------------| 1.4135948003
    epoch #88   |XXXXXXXXXX----------------------------------------| 1.41288434626
    epoch #89   |XXXXXXXXXX----------------------------------------| 1.41229595529
    epoch #90   |XXXXXXXXXX----------------------------------------| 1.4118076761
    epoch #91   |XXXXXXXXXX----------------------------------------| 1.41140232384
    epoch #92   |XXXXXXXXXX----------------------------------------| 1.41106623899
    epoch #93   |XXXXXXXXXX----------------------------------------| 1.41078841699
    epoch #94   |XXXXXXXXXX----------------------------------------| 1.41055988557
    epoch #95   |XXXXXXXXXX----------------------------------------| 1.41037325115
    epoch #96   |XXXXXXXXXX----------------------------------------| 1.41022236317
    epoch #97   |XXXXXXXXXX----------------------------------------| 1.41010206198
    epoch #98   |XXXXXXXXXX----------------------------------------| 1.41000798705
    epoch #99   |XXXXXXXXXX----------------------------------------| 1.40993642927

    Feeding a sequence to the network:
        Feeding the letter 'h', prediction: e [ 0.09909125  0.70012854  0.10116698  0.09961323]
        Feeding the letter 'e', prediction: l [ 0.09783366  0.09928262  0.70455709  0.09832664]
        Feeding the letter 'l', prediction: l [ 0.09794872  0.09866178  0.70439137  0.09899812]
        Feeding the letter 'l', prediction: o [ 0.09892499  0.09852191  0.09982541  0.70272769]


    """