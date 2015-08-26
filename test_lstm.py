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

    hiddenSize = 5
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

    #A small dataset, the word 'hello': 
    seq = [[1.,0.,0.,0.], [0.,1.,0.,0.], [0.,0.,1.,0.], [0.,0.,1.,0.], [0.,0.,0.,1.]]
    seq = [np.array(x) for x in seq]
    classes = [1, 2, 2, 3, 0]

    #Evaluation function: perform one epoch 
    def feval(x):

        myNet.reset_grad_param()
        errSum = 0.

        #FORWARD:
        outs, _ = myNet.forward(seq, [np.zeros(hiddenSize), np.zeros(hiddenSize)])

        errors = [CEErr.forward(outs[i], classes[i]) for i in xrange(4)]
        errSum = sum(errors)/4.0

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
    optimConf = {'learningRate':0.05, 'decayRate':0.99}
    optimState = {} #Just a container to save the optimization related variables (e.g. the previous gradient...)

    for it in xrange(3000):
        loss = RMSprop(feval, params, optimConf, optimState)
        maxLoss = max(loss, maxLoss)
        print "epoch #"+str(it)+"\t"+get_progress_bar(loss, maxLoss)+' '+str(loss)
        if loss < .5e-4:
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
        outputs ['outLinear', 'outLinear_t1', 'outLinear_t2', 'outLinear_t3']


    Checking the gradient computation ...  gradient OK.


    Training ...

    epoch #0    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX| 2.85481476519
    epoch #1    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--------| 2.42921455667
    epoch #2    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------------| 1.938155656
    epoch #3    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------------------| 1.68430090404
    epoch #4    |XXXXXXXXXXXXXXXXXXXXXXXXXXX-----------------------| 1.54704259982
    epoch #5    |XXXXXXXXXXXXXXXXXXXXXXXXX-------------------------| 1.45546717557
    epoch #6    |XXXXXXXXXXXXXXXXXXXXXXXX--------------------------| 1.38310835603
    epoch #7    |XXXXXXXXXXXXXXXXXXXXXXX---------------------------| 1.32338599428
    epoch #8    |XXXXXXXXXXXXXXXXXXXXXX----------------------------| 1.27350756042
    epoch #9    |XXXXXXXXXXXXXXXXXXXXX-----------------------------| 1.23156100073
    epoch #10   |XXXXXXXXXXXXXXXXXXXX------------------------------| 1.19617019417
    epoch #11   |XXXXXXXXXXXXXXXXXXXX------------------------------| 1.16632024512
    epoch #12   |XXXXXXXXXXXXXXXXXXX-------------------------------| 1.14112449062
    epoch #13   |XXXXXXXXXXXXXXXXXXX-------------------------------| 1.11970321717
    epoch #14   |XXXXXXXXXXXXXXXXXXX-------------------------------| 1.10121466381
    epoch #15   |XXXXXXXXXXXXXXXXXXX-------------------------------| 1.08492509077
    epoch #16   |XXXXXXXXXXXXXXXXXX--------------------------------| 1.07024208829
    epoch #17   |XXXXXXXXXXXXXXXXXX--------------------------------| 1.05670978612
    epoch #18   |XXXXXXXXXXXXXXXXXX--------------------------------| 1.0439864552
    epoch #19   |XXXXXXXXXXXXXXXXXX--------------------------------| 1.03181903274
    epoch #20   |XXXXXXXXXXXXXXXXX---------------------------------| 1.02002081675
    epoch #21   |XXXXXXXXXXXXXXXXX---------------------------------| 1.00845392064
    epoch #22   |XXXXXXXXXXXXXXXXX---------------------------------| 0.997016176877
    epoch #23   |XXXXXXXXXXXXXXXXX---------------------------------| 0.985631623199
    epoch #24   |XXXXXXXXXXXXXXXXX---------------------------------| 0.974243686478
    epoch #25   |XXXXXXXXXXXXXXXX----------------------------------| 0.962810328851
    epoch #26   |XXXXXXXXXXXXXXXX----------------------------------| 0.951300595857
    epoch #27   |XXXXXXXXXXXXXXXX----------------------------------| 0.939692157656
    epoch #28   |XXXXXXXXXXXXXXXX----------------------------------| 0.9279695522
    epoch #29   |XXXXXXXXXXXXXXXX----------------------------------| 0.916122925969
    epoch #30   |XXXXXXXXXXXXXXX-----------------------------------| 0.904147129999
    epoch #31   |XXXXXXXXXXXXXXX-----------------------------------| 0.89204107257
    epoch #32   |XXXXXXXXXXXXXXX-----------------------------------| 0.87980726033
    epoch #33   |XXXXXXXXXXXXXXX-----------------------------------| 0.867451480648
    epoch #34   |XXXXXXXXXXXXXX------------------------------------| 0.854982592434
    epoch #35   |XXXXXXXXXXXXXX------------------------------------| 0.842412402112
    epoch #36   |XXXXXXXXXXXXXX------------------------------------| 0.829755606825
    epoch #37   |XXXXXXXXXXXXXX------------------------------------| 0.817029788013
    epoch #38   |XXXXXXXXXXXXXX------------------------------------| 0.80425543468
    epoch #39   |XXXXXXXXXXXXX-------------------------------------| 0.791455965939
    epoch #40   |XXXXXXXXXXXXX-------------------------------------| 0.778657706569
    epoch #41   |XXXXXXXXXXXXX-------------------------------------| 0.76588974935
    epoch #42   |XXXXXXXXXXXXX-------------------------------------| 0.753183619792
    epoch #43   |XXXXXXXXXXXX--------------------------------------| 0.740572654246
    epoch #44   |XXXXXXXXXXXX--------------------------------------| 0.728091026704
    epoch #45   |XXXXXXXXXXXX--------------------------------------| 0.715772426206
    epoch #46   |XXXXXXXXXXXX--------------------------------------| 0.70364849356
    epoch #47   |XXXXXXXXXXXX--------------------------------------| 0.691747244153
    epoch #48   |XXXXXXXXXXX---------------------------------------| 0.680091777096
    epoch #49   |XXXXXXXXXXX---------------------------------------| 0.668699543785
    epoch #50   |XXXXXXXXXXX---------------------------------------| 0.657582306819
    epoch #51   |XXXXXXXXXXX---------------------------------------| 0.646746717234
    epoch #52   |XXXXXXXXXXX---------------------------------------| 0.636195271805
    epoch #53   |XXXXXXXXXX----------------------------------------| 0.625927359155
    epoch #54   |XXXXXXXXXX----------------------------------------| 0.615940168009
    epoch #55   |XXXXXXXXXX----------------------------------------| 0.606229353635
    epoch #56   |XXXXXXXXXX----------------------------------------| 0.596789468228
    epoch #57   |XXXXXXXXXX----------------------------------------| 0.587614219515
    epoch #58   |XXXXXXXXXX----------------------------------------| 0.578696630941
    epoch #59   |XXXXXXXXX-----------------------------------------| 0.570029158455
    epoch #60   |XXXXXXXXX-----------------------------------------| 0.561603794614
    epoch #61   |XXXXXXXXX-----------------------------------------| 0.553412171746
    epoch #62   |XXXXXXXXX-----------------------------------------| 0.545445665213
    epoch #63   |XXXXXXXXX-----------------------------------------| 0.537695493504
    epoch #64   |XXXXXXXXX-----------------------------------------| 0.530152811289
    epoch #65   |XXXXXXXXX-----------------------------------------| 0.522808792569
    epoch #66   |XXXXXXXXX-----------------------------------------| 0.515654702406
    epoch #67   |XXXXXXXX------------------------------------------| 0.508681956887
    epoch #68   |XXXXXXXX------------------------------------------| 0.501882171765
    epoch #69   |XXXXXXXX------------------------------------------| 0.49524720072
    epoch #70   |XXXXXXXX------------------------------------------| 0.488769164435
    epoch #71   |XXXXXXXX------------------------------------------| 0.482440471701
    epoch #72   |XXXXXXXX------------------------------------------| 0.476253833792
    epoch #73   |XXXXXXXX------------------------------------------| 0.470202273211
    epoch #74   |XXXXXXXX------------------------------------------| 0.464279127805
    epoch #75   |XXXXXXXX------------------------------------------| 0.458478051139
    epoch #76   |XXXXXXX-------------------------------------------| 0.452793009851
    epoch #77   |XXXXXXX-------------------------------------------| 0.447218278643
    epoch #78   |XXXXXXX-------------------------------------------| 0.441748433437
    epoch #79   |XXXXXXX-------------------------------------------| 0.436378343133
    epoch #80   |XXXXXXX-------------------------------------------| 0.431103160331
    epoch #81   |XXXXXXX-------------------------------------------| 0.425918311341
    epoch #82   |XXXXXXX-------------------------------------------| 0.420819485694
    epoch #83   |XXXXXXX-------------------------------------------| 0.415802625381
    epoch #84   |XXXXXXX-------------------------------------------| 0.410863913965
    epoch #85   |XXXXXXX-------------------------------------------| 0.405999765703
    epoch #86   |XXXXXXX-------------------------------------------| 0.401206814772
    epoch #87   |XXXXXX--------------------------------------------| 0.396481904694
    epoch #88   |XXXXXX--------------------------------------------| 0.391822077998
    epoch #89   |XXXXXX--------------------------------------------| 0.387224566198
    epoch #90   |XXXXXX--------------------------------------------| 0.382686780086
    epoch #91   |XXXXXX--------------------------------------------| 0.378206300394
    epoch #92   |XXXXXX--------------------------------------------| 0.373780868827
    epoch #93   |XXXXXX--------------------------------------------| 0.36940837947
    epoch #94   |XXXXXX--------------------------------------------| 0.365086870588
    epoch #95   |XXXXXX--------------------------------------------| 0.360814516804
    epoch #96   |XXXXXX--------------------------------------------| 0.356589621654
    epoch #97   |XXXXXX--------------------------------------------| 0.352410610515
    epoch #98   |XXXXXX--------------------------------------------| 0.348276023886
    epoch #99   |XXXXXX--------------------------------------------| 0.344184511023
    epoch #100  |XXXXX---------------------------------------------| 0.340134823897
    epoch #101  |XXXXX---------------------------------------------| 0.336125811474
    epoch #102  |XXXXX---------------------------------------------| 0.332156414302
    epoch #103  |XXXXX---------------------------------------------| 0.328225659376
    epoch #104  |XXXXX---------------------------------------------| 0.324332655282
    epoch #105  |XXXXX---------------------------------------------| 0.320476587591
    epoch #106  |XXXXX---------------------------------------------| 0.316656714494
    epoch #107  |XXXXX---------------------------------------------| 0.312872362666
    epoch #108  |XXXXX---------------------------------------------| 0.309122923334
    epoch #109  |XXXXX---------------------------------------------| 0.305407848545
    epoch #110  |XXXXX---------------------------------------------| 0.301726647615
    epoch #111  |XXXXX---------------------------------------------| 0.298078883749
    epoch #112  |XXXXX---------------------------------------------| 0.294464170815
    epoch #113  |XXXXX---------------------------------------------| 0.290882170264
    epoch #114  |XXXXX---------------------------------------------| 0.287332588185
    epoch #115  |XXXX----------------------------------------------| 0.283815172482
    epoch #116  |XXXX----------------------------------------------| 0.280329710163
    epoch #117  |XXXX----------------------------------------------| 0.276876024736
    epoch #118  |XXXX----------------------------------------------| 0.273453973698
    epoch #119  |XXXX----------------------------------------------| 0.270063446105
    epoch #120  |XXXX----------------------------------------------| 0.266704360233
    epoch #121  |XXXX----------------------------------------------| 0.263376661303
    epoch #122  |XXXX----------------------------------------------| 0.26008031927
    epoch #123  |XXXX----------------------------------------------| 0.256815326677
    epoch #124  |XXXX----------------------------------------------| 0.253581696564
    epoch #125  |XXXX----------------------------------------------| 0.250379460423
    epoch #126  |XXXX----------------------------------------------| 0.247208666202
    epoch #127  |XXXX----------------------------------------------| 0.244069376358
    epoch #128  |XXXX----------------------------------------------| 0.240961665939
    epoch #129  |XXXX----------------------------------------------| 0.23788562072
    epoch #130  |XXXX----------------------------------------------| 0.234841335367
    epoch #131  |XXXX----------------------------------------------| 0.231828911643
    epoch #132  |XXXX----------------------------------------------| 0.228848456657
    epoch #133  |XXX-----------------------------------------------| 0.225900081146
    epoch #134  |XXX-----------------------------------------------| 0.222983897804
    epoch #135  |XXX-----------------------------------------------| 0.220100019651
    epoch #136  |XXX-----------------------------------------------| 0.217248558453
    epoch #137  |XXX-----------------------------------------------| 0.214429623188
    epoch #138  |XXX-----------------------------------------------| 0.211643318568
    epoch #139  |XXX-----------------------------------------------| 0.208889743618
    epoch #140  |XXX-----------------------------------------------| 0.206168990319
    epoch #141  |XXX-----------------------------------------------| 0.203481142313
    epoch #142  |XXX-----------------------------------------------| 0.200826273682
    epoch #143  |XXX-----------------------------------------------| 0.1982044478
    epoch #144  |XXX-----------------------------------------------| 0.195615716266
    epoch #145  |XXX-----------------------------------------------| 0.193060117919
    epoch #146  |XXX-----------------------------------------------| 0.190537677942
    epoch #147  |XXX-----------------------------------------------| 0.188048407048
    epoch #148  |XXX-----------------------------------------------| 0.18559230077
    epoch #149  |XXX-----------------------------------------------| 0.183169338838
    epoch #150  |XXX-----------------------------------------------| 0.180779484653
    epoch #151  |XXX-----------------------------------------------| 0.17842268486
    epoch #152  |XXX-----------------------------------------------| 0.176098869015
    epoch #153  |XXX-----------------------------------------------| 0.173807949353
    epoch #154  |XXX-----------------------------------------------| 0.171549820647
    epoch #155  |XX------------------------------------------------| 0.169324360164
    epoch #156  |XX------------------------------------------------| 0.167131427703
    epoch #157  |XX------------------------------------------------| 0.164970865736
    epoch #158  |XX------------------------------------------------| 0.162842499614
    epoch #159  |XX------------------------------------------------| 0.160746137866
    epoch #160  |XX------------------------------------------------| 0.158681572565
    epoch #161  |XX------------------------------------------------| 0.156648579764
    epoch #162  |XX------------------------------------------------| 0.154646919998
    epoch #163  |XX------------------------------------------------| 0.152676338837
    epoch #164  |XX------------------------------------------------| 0.150736567495
    epoch #165  |XX------------------------------------------------| 0.148827323484
    epoch #166  |XX------------------------------------------------| 0.146948311304
    epoch #167  |XX------------------------------------------------| 0.145099223166
    epoch #168  |XX------------------------------------------------| 0.143279739742
    epoch #169  |XX------------------------------------------------| 0.14148953094
    epoch #170  |XX------------------------------------------------| 0.139728256681
    epoch #171  |XX------------------------------------------------| 0.137995567699
    epoch #172  |XX------------------------------------------------| 0.136291106339
    epoch #173  |XX------------------------------------------------| 0.134614507354
    epoch #174  |XX------------------------------------------------| 0.132965398696
    epoch #175  |XX------------------------------------------------| 0.131343402298
    epoch #176  |XX------------------------------------------------| 0.129748134849
    epoch #177  |XX------------------------------------------------| 0.128179208539
    epoch #178  |XX------------------------------------------------| 0.126636231801
    epoch #179  |XX------------------------------------------------| 0.125118810013
    epoch #180  |XX------------------------------------------------| 0.123626546192
    epoch #181  |XX------------------------------------------------| 0.122159041651
    epoch #182  |XX------------------------------------------------| 0.120715896631
    epoch #183  |XX------------------------------------------------| 0.119296710907
    epoch #184  |XX------------------------------------------------| 0.117901084356
    epoch #185  |XX------------------------------------------------| 0.116528617507
    epoch #186  |XX------------------------------------------------| 0.115178912046
    epoch #187  |X-------------------------------------------------| 0.113851571299
    epoch #188  |X-------------------------------------------------| 0.112546200682
    epoch #189  |X-------------------------------------------------| 0.111262408121
    epoch #190  |X-------------------------------------------------| 0.109999804434
    epoch #191  |X-------------------------------------------------| 0.108758003699
    epoch #192  |X-------------------------------------------------| 0.107536623574
    epoch #193  |X-------------------------------------------------| 0.106335285605
    epoch #194  |X-------------------------------------------------| 0.105153615498
    epoch #195  |X-------------------------------------------------| 0.103991243365
    epoch #196  |X-------------------------------------------------| 0.102847803949
    epoch #197  |X-------------------------------------------------| 0.101722936821
    epoch #198  |X-------------------------------------------------| 0.100616286558
    epoch #199  |X-------------------------------------------------| 0.099527502892

    Training over.

    Feeding a sequence to the network:
        Feeding the letter 'h', prediction: e [ 0.00558533  0.85767486  0.13036006  0.00637975]
        Feeding the letter 'e', prediction: l [ 0.00112999  0.03024236  0.96225819  0.00636946]
        Feeding the letter 'l', prediction: l [ 0.00236831  0.00348877  0.8795488   0.11459412]
        Feeding the letter 'l', prediction: o [ 0.0057386   0.00164283  0.03535152  0.95726705]


  
    """