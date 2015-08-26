#!/usr/bin/python

import numpy as np
from pyNNGraph import *
import cPickle as pkl
from reader import *

def get_progress_bar(loss, maxLoss, scale=50):
    n = int(scale*loss/maxLoss)
    s = '|'+'X'*n+(scale-n)*'-'+'|'
    return s

def clip_gradient(gradParams, maxValue=1.):
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
                'outDropout'+str(i):Dropout(layersSize, 0.5)
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

def get_error(net, sequence):
    """"""
    net.training_mode_OFF()
    net.reset_recurrent_states()
    errSum = 0.
    idx = 0
    while idx+1 < len(sequence):
        outs = net.sequence_prediction([sequence[idx]])
        target = sequence[idx+1]
        errSum += CEErr.forward(outs[0], target)
        idx += 1
    net.training_mode_ON()
    return errSum/len(sequence)

def get_prediction(net, seed, numChars, temperature):
    """"""
    net.training_mode_OFF()
    net.reset_recurrent_states()
    prediction = list(seed)
    for char in list(seed):
        outs = net.sequence_prediction([dataset.dict[char]])
        probs = CEErr.get_probabilities(np.divide(outs[0], temperature))
        nextChar = dataset.inv_dict[np.argmax(np.random.multinomial(1,probs))]
    for _ in xrange(numChars):
        outs = net.sequence_prediction([dataset.dict[nextChar]])
        probs = CEErr.get_probabilities(np.divide(outs[0], temperature))
        nextChar = dataset.inv_dict[np.argmax(np.random.multinomial(1,probs))]
        prediction.append(nextChar)
    net.training_mode_ON()
    return ''.join(prediction)

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
    NUM_LAYERS = 2
    HIDDEN_SIZE = 300
    MINIBATCH_SIZE = 50
    CHECK_VAL_EVERY = 300 #Check the validation error every 20 epochs
    SAVE_AS = "languageModel_LSTM_warAndPeace.pkl"

    #Fetching the data
    with open('./dataset_WAndP.pkl', 'rb') as inStream:
        dataset = pkl.load(inStream)

    trainSet = dataset.trainSet
    validSet = dataset.validSet

    inputSize = dataset.get_vocabulary_size()

    print "\nTraining set size:", len(trainSet)
    print "Validation set size:", len(validSet)
    print ""

    #Creating the network
    myNet = build_network(NUM_LAYERS, inputSize, HIDDEN_SIZE)
    CEErr = CELayer(inputSize)

    #get lists of references to parameters of the network
    params, gradParams = myNet.get_link_to_parameters()

    myNet.unwrap(SEQ_SIZE) #unwrap the network on 50 timesteps 

    dataInc = 0
    Hins = [np.zeros(HIDDEN_SIZE) for _ in xrange(2*NUM_LAYERS)] #initial Hidden states

    myNet.training_mode_ON()

    #Evaluation function: perform one epoch 
    def feval(x):

        myNet.reset_grad_param()
        errSum = 0.
        global dataInc
        global Hins

        for i in xrange(MINIBATCH_SIZE):

            currentSequence = trainSet[dataInc:dataInc+SEQ_SIZE]
            currentTarget = trainSet[dataInc+1:dataInc+SEQ_SIZE+1]
            dataInc += SEQ_SIZE

            #FORWARD:
            outs, Hins = myNet.forward(currentSequence, Hins)

            errors = [CEErr.forward(outs[i], currentTarget[i]) for i in xrange(SEQ_SIZE)]
            errSum += sum(errors)

            #BACKWARD:
            gradOutputs = [CEErr.backward(outs[i], currentTarget[i]) for i in xrange(SEQ_SIZE)]
            myNet.backward(currentSequence, gradOutputs)

            if dataInc+SEQ_SIZE >= len(trainSet):
                dataInc = 0
                Hins = [np.zeros(HIDDEN_SIZE) for _ in xrange(2*NUM_LAYERS)]

        clip_gradient(gradParams)
        errSum /= float(MINIBATCH_SIZE)*float(SEQ_SIZE)

        return errSum, gradParams

    
    maxLoss = 0.
    minValidLoss = 1e+10 

    #Training routine:
    optimConf = {'learningRate':0.008, 'decayRate':0.95}
    optimState = {} 

    print get_prediction(myNet, "The ", 500, .9)
    for it in xrange(1, 50000):
        loss = RMSprop(feval, params, optimConf, optimState)
        maxLoss = max(loss, maxLoss)
        print "epoch #"+str(it)+"\t"+get_progress_bar(loss, maxLoss)+' '+str(loss)
        if it == 5000:  #basic learning rate decay
            optimConf['learningRate'] = 0.004
        if it == 10000:
            optimConf['learningRate'] = 0.001
        if it % CHECK_VAL_EVERY == 0: #compute the validation error
            validError = get_error(myNet, validSet)
            minValidLoss = min(minValidLoss, validError)
            print "\t\tValidation error:", validError,
            if validError == minValidLoss:
                print " New best validation error ... saving."
                myNet.save(SAVE_AS)
            else:
                print ""
            print get_prediction(myNet, "The ", 500, .9)
        if loss < 1e-2:
            print "\nTraining over."
            break
    

