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
    """In this example we are going to build a character-based language model:

        let V be the size of the input (number of characters in our vocabulary).
        let H be the size of the hidden layer. 
        We are going to build a LSTM neural network, one LSTM layer looks like this:
        

                            hiddenOut_t
                                 ^
                                 |
                 ---------->CMulTable(H)
                 |               ^                                
                 |               |
                 |            Tanh(H)
                 |               ^
                 |               |
                 |               |
                 |               | c_t            #c is the carousel value 
                 |   c_{t-1}     |
                 |      |    CAddTable(H)
                 |      |     ^       ^
                 |      v     |       |
                 |  CMulTable(H)      |
                 |       ^            |
                 |       |            |
                 |       |       CMulTable(H)
                 |       |       ^        ^
                 |       |       |        |          
               Sgmd(H) Sgmd(H) Sgmd(H)  Tanh(H)  #output_gate forget_gate input_gate forward_signal 
                 ^       ^       ^        ^
                 |       |       |        |
                SplitTable(4*H,[H, H, H, H])
                             ^
                             |
                 ----->CAddTable(4*H)<-----
                 |                        |
            Linear(V,4*H)            Linear(H,4*H)
                 ^                        ^
                 |                        |
             charIn_t               hiddenOut_{t-1}


        We can see there are two recurrent connexions, one for the constant error carousel c_t,
        and another one feeding the output of the layer to its input.
           
    """

    hiddenSize = 3
    InputSize = 4

    #Allocating all the nodes we need
    CEErr = CELayer(InputSize)

    nodesTable = {
                    'frwdLinear':Linear(InputSize, 4*hiddenSize), 
                    'recLinear':Linear(hiddenSize, 4*hiddenSize), 
                    'add1':CAddTable(4*hiddenSize), 
                    'add2':CAddTable(hiddenSize),
                    'split':SplitTable(4*hiddenSize, [hiddenSize, hiddenSize, hiddenSize, hiddenSize]), 
                    'outGate_sgmd':Sigmoid(hiddenSize), 
                    'inGate_sgmd':Sigmoid(hiddenSize),
                    'frgtGate_sgmd':Sigmoid(hiddenSize),
                    'frwd_tanh':Tanh(hiddenSize),
                    'in_fwd_mul':CMulTable(hiddenSize),
                    'frgt_c_mul':CMulTable(hiddenSize),
                    'out_tanh':Tanh(hiddenSize),
                    'out_out_mul':CMulTable(hiddenSize),
                    'outLinear':Linear(hiddenSize, InputSize)#,
                    #'outDropout':Dropout(InputSize)
                    }

    inputNodes = ['frwdLinear']
    outputNodes = ['outLinear']

    evaluationSequence = ['recLinear', 
                          'add1', 
                          'split', 
                          'outGate_sgmd', 
                          'inGate_sgmd', 
                          'frgtGate_sgmd', 
                          'frwd_tanh', 
                          'in_fwd_mul', 
                          'frgt_c_mul', 
                          'add2', 
                          'out_tanh', 
                          'out_out_mul', 
                          'outLinear'
                         ]

    myNet = Network(nodesTable, inputNodes, outputNodes, evaluationSequence)

    #Creating the network
    myNet.link_nodes('frwdLinear', 'add1')
    myNet.link_nodes('recLinear', 'add1')
    myNet.link_nodes('add1', 'split')
    myNet.link_nodes('split', 'outGate_sgmd')
    myNet.link_nodes('split', 'frgtGate_sgmd')
    myNet.link_nodes('split', 'inGate_sgmd')
    myNet.link_nodes('split', 'frwd_tanh')
    myNet.link_nodes('outGate_sgmd', 'out_out_mul')
    myNet.link_nodes('frgtGate_sgmd', 'frgt_c_mul')
    myNet.link_nodes('inGate_sgmd', 'in_fwd_mul')
    myNet.link_nodes('frwd_tanh', 'in_fwd_mul')
    myNet.link_nodes('in_fwd_mul', 'add2')
    myNet.link_nodes('frgt_c_mul', 'add2')
    myNet.link_nodes('add2', 'out_tanh')
    myNet.link_nodes('out_tanh', 'out_out_mul')
    myNet.link_nodes('out_out_mul', 'outLinear')

    myNet.recurrent_connexion('add2', 'frgt_c_mul') 
    myNet.recurrent_connexion('out_out_mul', 'recLinear') 

    #get lists of references to parameters of the network
    params, gradParams = myNet.get_link_to_parameters()

    myNet.unwrap(4) #unwrap the network on 4 timesteps 

    print "\nNetwork unwrapped, \n\tinputs:",myNet.inputNodes,"\n\toutputs",myNet.outputNodes

    print myNet.nodesTable.keys()

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
    optimConf = {'learningRate':0.1, 'learningRateDecay':0.001, 'momentum':0.5, 'weightDecay':0.0}
    optimState = {} #Just a container to save the optimization related variables (e.g. the previous gradient...)

    for it in xrange(1000):
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

  
    """