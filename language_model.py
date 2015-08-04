#!/usr/bin/python

import numpy as np
from pyNNGraph import *
import reader as rd

def get_progress_bar(loss, maxLoss, scale=50):
    n = int(scale*loss/maxLoss)
    s = '|'+'X'*n+(scale-n)*'-'+'|'
    return s

def clip_gradient(gradParams, maxValue=5.):
    """"""
    for gradParam in gradParams:
        gradParam.clip(-maxValue,maxValue,gradParam)

def build_network(numLayers, inputSize, layersSize):
    """"""
    inputNodes = ['frwdLinear0']
    evaluationSequence = []
    nodesTable = {}
    for i in xrange(numLayers):
        nodesTable.update({
                'frwdLinear'+str(i):SparseLinear(inputSize, 4*layersSize) if i==0 else Linear(layersSize, 4*layersSize), 
                'recLinear'+str(i):Linear(layersSize, 4*layersSize), 
                'add1'+str(i):CAddTable(4*layersSize), 
                'add2'+str(i):CAddTable(layersSize),
                'split'+str(i):SplitTable(4*layersSize, [layersSize, layersSize, layersSize, layersSize]), 
                'outGate_sgmd'+str(i):Sigmoid(layersSize), 
                'inGate_sgmd'+str(i):Sigmoid(layersSize),
                'frgtGate_sgmd'+str(i):Sigmoid(layersSize),
                'frwd_tanh'+str(i):Tanh(layersSize),
                'in_fwd_mul'+str(i):CMulTable(layersSize),
                'frgt_c_mul'+str(i):CMulTable(layersSize),
                'out_tanh'+str(i):Tanh(layersSize),
                'out_out_mul'+str(i):CMulTable(layersSize),
                'outDropout'+str(i):Dropout(layersSize, 0.3)
                })
        if i != 0:
            evaluationSequence += ['frwdLinear'+str(i)]
        evaluationSequence += ['recLinear'+str(i), 
                               'add1'+str(i), 
                               'split'+str(i), 
                               'outGate_sgmd'+str(i), 
                               'inGate_sgmd'+str(i), 
                               'frgtGate_sgmd'+str(i), 
                               'frwd_tanh'+str(i), 
                               'in_fwd_mul'+str(i), 
                               'frgt_c_mul'+str(i), 
                               'add2'+str(i), 
                               'out_tanh'+str(i), 
                               'out_out_mul'+str(i),
                               'outDropout'+str(i)
                              ] 

    evaluationSequence += ['outLinear']
    nodesTable.update({'outLinear': Linear(layersSize, inputSize)})
    outputNodes = ['outLinear']

    myNet = Network(nodesTable, inputNodes, outputNodes, evaluationSequence)

    for i in xrange(numLayers):
        if i != 0:
            #myNet.link_nodes('out_out_mul'+str(i-1), 'frwdLinear'+str(i))
            myNet.link_nodes('outDropout'+str(i-1), 'frwdLinear'+str(i))
        myNet.link_nodes('frwdLinear'+str(i), 'add1'+str(i))
        myNet.link_nodes('recLinear'+str(i), 'add1'+str(i))
        myNet.link_nodes('add1'+str(i), 'split'+str(i))
        myNet.link_nodes('split'+str(i), 'outGate_sgmd'+str(i))
        myNet.link_nodes('split'+str(i), 'frgtGate_sgmd'+str(i))
        myNet.link_nodes('split'+str(i), 'inGate_sgmd'+str(i))
        myNet.link_nodes('split'+str(i), 'frwd_tanh'+str(i))
        myNet.link_nodes('outGate_sgmd'+str(i), 'out_out_mul'+str(i))
        myNet.link_nodes('frgtGate_sgmd'+str(i), 'frgt_c_mul'+str(i))
        myNet.link_nodes('inGate_sgmd'+str(i), 'in_fwd_mul'+str(i))
        myNet.link_nodes('frwd_tanh'+str(i), 'in_fwd_mul'+str(i))
        myNet.link_nodes('in_fwd_mul'+str(i), 'add2'+str(i))
        myNet.link_nodes('frgt_c_mul'+str(i), 'add2'+str(i))
        myNet.link_nodes('add2'+str(i), 'out_tanh'+str(i))
        myNet.link_nodes('out_tanh'+str(i), 'out_out_mul'+str(i))
        myNet.link_nodes('out_out_mul'+str(i), 'outDropout'+str(i))

        myNet.recurrent_connexion('add2'+str(i), 'frgt_c_mul'+str(i)) 
        myNet.recurrent_connexion('out_out_mul'+str(i), 'recLinear'+str(i)) 

    #myNet.link_nodes('out_out_mul'+str(numLayers-1), 'outLinear')
    myNet.link_nodes('outDropout'+str(numLayers-1), 'outLinear')

    return myNet

def get_error(net, data):
    """"""
    errSum = 0
    for seq, target in data:
        outs = myNet.forward(seq)
        errors = [CEErr.forward(outs[i], target[i]) for i in xrange(SEQ_SIZE)]
        errSum += sum(errors)
    errSum /= float(SEQ_SIZE)*len(data)
    return errSum


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

    SEQ_SIZE = 50
    STEP_SIZE = 30
    MINIBATCH_SIZE = 40
    CHECK_VAL_EVERY = 20 #Check the validation error every 20 epochs
    SAVE_AS = "languageModel_LSTM.pkl"

    #Fetching the data
    dataset = rd.Reader('data/lovecraft.txt', SEQ_SIZE, STEP_SIZE)
    inputSize = dataset.get_vocabulary_size()

    data = [(seq,target) for seq, target in zip(dataset.sequences, dataset.targets)]
    np.random.shuffle(data)

    split = int(0.95*len(data))
    trainSet = data[:split]
    validSet = data[split:]

    del data

    print "\nTraining set size:", len(trainSet)
    print "Validation set size:", len(validSet)
    print ""

    #Creating the network
    myNet = build_network(2, inputSize, 100)
    CEErr = CELayer(inputSize)

    #get lists of references to parameters of the network
    params, gradParams = myNet.get_link_to_parameters()

    myNet.unwrap(50) #unwrap the network on 50 timesteps 

    dataInc = 0

    myNet.training_mode_ON()

    #Evaluation function: perform one epoch 
    def feval(x):

        myNet.reset_grad_param()
        errSum = 0.
        global dataInc

        for i in xrange(MINIBATCH_SIZE):

            currentIdx = (dataInc + i)%len(trainSet)
            currentSequence = trainSet[currentIdx][0]
            currenttarget = trainSet[currentIdx][1]

            #FORWARD:
            outs = myNet.forward(currentSequence)

            errors = [CEErr.forward(outs[i], currenttarget[i]) for i in xrange(SEQ_SIZE)]
            errSum += sum(errors)

            #BACKWARD:
            gradOutputs = [CEErr.backward(outs[i], currenttarget[i]) for i in xrange(SEQ_SIZE)]
            myNet.backward(currentSequence, gradOutputs)

        clip_gradient(gradParams)
        dataInc += MINIBATCH_SIZE
        errSum /= float(MINIBATCH_SIZE)*float(SEQ_SIZE)

        return errSum, gradParams

    
    maxLoss = 0.
    minValidLoss = 1e+10 

    #Training routine:
    optimConf = {'learningRate':0.002, 'decayRate':0.99}
    optimState = {} #Just a container to save the optimization related variables (e.g. the previous gradient...)

    for it in xrange(1, 10000):
        loss = RMSprop(feval, params, optimConf, optimState)
        maxLoss = max(loss, maxLoss)
        print "epoch #"+str(it)+"\t"+get_progress_bar(loss, maxLoss)+' '+str(loss)
        if it % CHECK_VAL_EVERY == 0: #compute the validation error
            myNet.training_mode_OFF()
            validError = get_error(myNet, validSet)
            myNet.training_mode_OFF()
            minValidLoss = min(minValidLoss, validError)
            print "\t\tValidation error:", validError,
            if validError == minValidLoss:
                print " New best validation error ... saving."
                myNet.save(SAVE_AS)
            else:
                print ""
        if loss < 1e-5:
            print "\nTraining over."
            break
    

    """ EXCPECTED OUTPUT:

  
    """
