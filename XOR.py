#!/usr/bin/python

import numpy as np
from pyNNGraph import *

def get_progress_bar(loss, maxLoss, scale=50):
    n = int(scale*loss/maxLoss)
    s = '|'+'X'*n+(scale-n)*'-'+'|'
    return s

if __name__ == "__main__":
    """To solve the XOR problem, let's build the following net:

        Xin(2) --->Linear(2,2)--->Sigmoid(2)--->Linear(2,1)--->Tanh(1)---> Out(1)

    """

    #Allocating all the nodes we need
    linear1 = Linear(2, 2)
    linear2 = Linear(2, 1)
    tanh = Tanh(1)
    sigmoid = Sigmoid(2)

    #We will use MSE as error metric
    err1 = MSELayer(1)

    nodesTable = {'linear1':linear1, 'sigmoid':sigmoid, 'tanh':tanh, 'linear2':linear2}

    inputNodes={'linear1'}
    outputNodes={'tanh'}
    evaluationSequence=['sigmoid', 'linear2', 'tanh']

    myNet = Network(nodesTable, inputNodes, outputNodes, evaluationSequence)

    #Creating the network
    myNet.link_nodes('linear1', 'sigmoid')
    myNet.link_nodes('sigmoid', 'linear2')
    myNet.link_nodes('linear2', 'tanh')

    #A very small dataset
    data = [(0.,0.), (1.,0.), (1.,1.), (0.,1.)]
    target = [(1.), (-1.), (1.), (-1.)]

    #get lists of references to parameters of the network
    params, gradParams = myNet.get_link_to_parameters()

    #Evaluation function: perform one epoch 
    def feval(x):

        myNet.reset_grad_param()

        errSum = 0.
        for cx,ct in zip(data, target): 

            currentInput = np.array(cx)
            currentTarget = np.array(ct)

            outs = myNet.forward([currentInput]) #Push the input inside the network
            errSum += err1.forward(outs[0], currentTarget)
            gradOutput1 = err1.backward(outs[0], currentTarget)

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
    optimConf = {'learningRate':0.2, 'learningRateDecay':0.0, 'momentum':0.2, 'weightDecay':0.}
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
    for cx,ct in zip(data, target): 
        currentInput = np.array(cx)
        currentTarget = np.array(ct)
        outs = myNet.forward([currentInput])
        print "\tinput",cx,"yields output",outs[0]

    print ""


    """ EXCPECTED OUTPUT:


    Checking the gradient computation ...  gradient OK.


    Training ...

    epoch #0    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX| [ 5.16666604]
    epoch #1    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--------| [ 4.36862361]
    epoch #2    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------| [ 4.08996359]
    epoch #3    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------| [ 4.17526598]
    epoch #4    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------| [ 4.21336128]
    epoch #5    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------| [ 4.26446679]
    epoch #6    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------| [ 4.29807883]
    epoch #7    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------| [ 4.31873012]
    epoch #8    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------| [ 4.32250659]
    epoch #9    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------| [ 4.31939504]
    epoch #10   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------| [ 4.30895541]
    epoch #11   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------| [ 4.29736474]
    epoch #12   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------| [ 4.28379783]
    epoch #13   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------| [ 4.27016649]
    epoch #14   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------| [ 4.25685575]
    epoch #15   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------| [ 4.2430874]
    epoch #16   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------| [ 4.23079244]
    epoch #17   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------| [ 4.21741313]
    epoch #18   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------| [ 4.20619249]
    epoch #19   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------| [ 4.19332508]
    epoch #20   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------| [ 4.18306297]
    epoch #21   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------| [ 4.17070257]
    epoch #22   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------| [ 4.1612497]
    epoch #23   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------| [ 4.1493463]
    epoch #24   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------| [ 4.14055721]
    epoch #25   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------| [ 4.12904258]
    epoch #26   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------| [ 4.12078407]
    epoch #27   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------| [ 4.10958112]
    epoch #28   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------| [ 4.10173234]
    epoch #29   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------| [ 4.09075836]
    epoch #30   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------| [ 4.08320884]
    epoch #31   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------| [ 4.07237642]
    epoch #32   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------| [ 4.06502383]
    epoch #33   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------| [ 4.05424092]
    epoch #34   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------| [ 4.04698922]
    epoch #35   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------| [ 4.03615885]
    epoch #36   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX------------| [ 4.02891695]
    epoch #37   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX------------| [ 4.01793689]
    epoch #38   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX------------| [ 4.01061802]
    epoch #39   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX------------| [ 3.99938058]
    epoch #40   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX------------| [ 3.9919023]
    epoch #41   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX------------| [ 3.98029422]
    epoch #42   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX------------| [ 3.97257909]
    epoch #43   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX------------| [ 3.96048156]
    epoch #44   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX------------| [ 3.95245841]
    epoch #45   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX------------| [ 3.93974719]
    epoch #46   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX------------| [ 3.93135287]
    epoch #47   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-------------| [ 3.9178983]
    epoch #48   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-------------| [ 3.9090796]
    epoch #49   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-------------| [ 3.8947463]
    epoch #50   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-------------| [ 3.88546168]
    epoch #51   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-------------| [ 3.87010778]
    epoch #52   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-------------| [ 3.86032816]
    epoch #53   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-------------| [ 3.84380393]
    epoch #54   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-------------| [ 3.83351205]
    epoch #55   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--------------| [ 3.81565816]
    epoch #56   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--------------| [ 3.80484576]
    epoch #57   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--------------| [ 3.78549212]
    epoch #58   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--------------| [ 3.77415467]
    epoch #59   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--------------| [ 3.75312161]
    epoch #60   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--------------| [ 3.74125109]
    epoch #61   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------------| [ 3.71835537]
    epoch #62   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------------| [ 3.70593233]
    epoch #63   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------------| [ 3.68100082]
    epoch #64   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------------| [ 3.66798815]
    epoch #65   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------------| [ 3.6408805]
    epoch #66   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------------| [ 3.62722156]
    epoch #67   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------| [ 3.59786034]
    epoch #68   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------| [ 3.58348426]
    epoch #69   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------| [ 3.55188687]
    epoch #70   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------| [ 3.53672268]
    epoch #71   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------------| [ 3.50302565]
    epoch #72   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------------| [ 3.48702502]
    epoch #73   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------------| [ 3.45149064]
    epoch #74   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------------| [ 3.43465693]
    epoch #75   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX------------------| [ 3.39765512]
    epoch #76   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX------------------| [ 3.38007448]
    epoch #77   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX------------------| [ 3.34203983]
    epoch #78   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX------------------| [ 3.32391029]
    epoch #79   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-------------------| [ 3.28528318]
    epoch #80   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-------------------| [ 3.26694292]
    epoch #81   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-------------------| [ 3.22811133]
    epoch #82   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-------------------| [ 3.21007898]
    epoch #83   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--------------------| [ 3.17134191]
    epoch #84   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--------------------| [ 3.15440086]
    epoch #85   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--------------------| [ 3.11597316]
    epoch #86   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--------------------| [ 3.10135602]
    epoch #87   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------------------| [ 3.06342495]
    epoch #88   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------------------| [ 3.05317519]
    epoch #89   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------------------| [ 3.01598076]
    epoch #90   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------------------| [ 3.01353385]
    epoch #91   |XXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------------| [ 2.97729762]
    epoch #92   |XXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------------| [ 2.98802156]
    epoch #93   |XXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------------| [ 2.95213279]
    epoch #94   |XXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------------| [ 2.98252524]
    epoch #95   |XXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------------| [ 2.94301012]
    epoch #96   |XXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------------| [ 2.99583472]
    epoch #97   |XXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------------| [ 2.94270822]
    epoch #98   |XXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------------------| [ 3.0086597]
    epoch #99   |XXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------------| [ 2.93246576]
    epoch #100  |XXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------------| [ 2.98969157]
    epoch #101  |XXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------------| [ 2.89807222]
    epoch #102  |XXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------------| [ 2.92425851]
    epoch #103  |XXXXXXXXXXXXXXXXXXXXXXXXXXX-----------------------| [ 2.84159606]
    epoch #104  |XXXXXXXXXXXXXXXXXXXXXXXXXXX-----------------------| [ 2.82129633]
    epoch #105  |XXXXXXXXXXXXXXXXXXXXXXXXXX------------------------| [ 2.77097726]
    epoch #106  |XXXXXXXXXXXXXXXXXXXXXXXXXX------------------------| [ 2.6922221]
    epoch #107  |XXXXXXXXXXXXXXXXXXXXXXXXXX------------------------| [ 2.68844231]
    epoch #108  |XXXXXXXXXXXXXXXXXXXXXXXX--------------------------| [ 2.53969787]
    epoch #109  |XXXXXXXXXXXXXXXXXXXXXXXXX-------------------------| [ 2.58780691]
    epoch #110  |XXXXXXXXXXXXXXXXXXXXXX----------------------------| [ 2.36244387]
    epoch #111  |XXXXXXXXXXXXXXXXXXXXXXX---------------------------| [ 2.45844666]
    epoch #112  |XXXXXXXXXXXXXXXXXXXX------------------------------| [ 2.16333891]
    epoch #113  |XXXXXXXXXXXXXXXXXXXXXX----------------------------| [ 2.29228776]
    epoch #114  |XXXXXXXXXXXXXXXXXX--------------------------------| [ 1.94994119]
    epoch #115  |XXXXXXXXXXXXXXXXXXXX------------------------------| [ 2.08629821]
    epoch #116  |XXXXXXXXXXXXXXXX----------------------------------| [ 1.72616579]
    epoch #117  |XXXXXXXXXXXXXXXXX---------------------------------| [ 1.83638051]
    epoch #118  |XXXXXXXXXXXXXX------------------------------------| [ 1.48565032]
    epoch #119  |XXXXXXXXXXXXXX------------------------------------| [ 1.53407183]
    epoch #120  |XXXXXXXXXXX---------------------------------------| [ 1.21727749]
    epoch #121  |XXXXXXXXXXX---------------------------------------| [ 1.18288672]
    epoch #122  |XXXXXXXX------------------------------------------| [ 0.92822744]
    epoch #123  |XXXXXXXX------------------------------------------| [ 0.83625493]
    epoch #124  |XXXXXX--------------------------------------------| [ 0.67674801]
    epoch #125  |XXXXX---------------------------------------------| [ 0.59700239]
    epoch #126  |XXXXX---------------------------------------------| [ 0.52725784]
    epoch #127  |XXXX----------------------------------------------| [ 0.48520193]
    epoch #128  |XXXX----------------------------------------------| [ 0.45312415]
    epoch #129  |XXXX----------------------------------------------| [ 0.42732648]
    epoch #130  |XXX-----------------------------------------------| [ 0.40468303]
    epoch #131  |XXX-----------------------------------------------| [ 0.38417113]
    epoch #132  |XXX-----------------------------------------------| [ 0.36534706]
    epoch #133  |XXX-----------------------------------------------| [ 0.34798784]
    epoch #134  |XXX-----------------------------------------------| [ 0.33193654]
    epoch #135  |XXX-----------------------------------------------| [ 0.31706369]
    epoch #136  |XX------------------------------------------------| [ 0.30325685]
    epoch #137  |XX------------------------------------------------| [ 0.29041692]
    epoch #138  |XX------------------------------------------------| [ 0.27845599]
    epoch #139  |XX------------------------------------------------| [ 0.26729575]
    epoch #140  |XX------------------------------------------------| [ 0.25686623]
    epoch #141  |XX------------------------------------------------| [ 0.24710477]
    epoch #142  |XX------------------------------------------------| [ 0.23795513]
    epoch #143  |XX------------------------------------------------| [ 0.22936673]
    epoch #144  |XX------------------------------------------------| [ 0.22129403]
    epoch #145  |XX------------------------------------------------| [ 0.21369592]
    epoch #146  |X-------------------------------------------------| [ 0.20653524]
    epoch #147  |X-------------------------------------------------| [ 0.19977835]
    epoch #148  |X-------------------------------------------------| [ 0.19339475]
    epoch #149  |X-------------------------------------------------| [ 0.18735674]
    epoch #150  |X-------------------------------------------------| [ 0.18163913]
    epoch #151  |X-------------------------------------------------| [ 0.17621897]
    epoch #152  |X-------------------------------------------------| [ 0.17107532]
    epoch #153  |X-------------------------------------------------| [ 0.16618904]
    epoch #154  |X-------------------------------------------------| [ 0.16154264]
    epoch #155  |X-------------------------------------------------| [ 0.15712008]
    epoch #156  |X-------------------------------------------------| [ 0.15290665]
    epoch #157  |X-------------------------------------------------| [ 0.14888882]
    epoch #158  |X-------------------------------------------------| [ 0.14505417]
    epoch #159  |X-------------------------------------------------| [ 0.14139124]
    epoch #160  |X-------------------------------------------------| [ 0.13788946]
    epoch #161  |X-------------------------------------------------| [ 0.13453908]
    epoch #162  |X-------------------------------------------------| [ 0.13133107]
    epoch #163  |X-------------------------------------------------| [ 0.1282571]
    epoch #164  |X-------------------------------------------------| [ 0.12530941]
    epoch #165  |X-------------------------------------------------| [ 0.12248083]
    epoch #166  |X-------------------------------------------------| [ 0.1197647]
    epoch #167  |X-------------------------------------------------| [ 0.1171548]
    epoch #168  |X-------------------------------------------------| [ 0.11464537]
    epoch #169  |X-------------------------------------------------| [ 0.11223104]
    epoch #170  |X-------------------------------------------------| [ 0.10990678]
    epoch #171  |X-------------------------------------------------| [ 0.10766792]
    epoch #172  |X-------------------------------------------------| [ 0.10551007]
    epoch #173  |X-------------------------------------------------| [ 0.10342915]
    epoch #174  |--------------------------------------------------| [ 0.10142131]
    epoch #175  |--------------------------------------------------| [ 0.09948298]

    Training over.

    Checking the output for each input:
        input (0.0, 0.0) yields output [ 0.83534278]
        input (1.0, 0.0) yields output [-0.80673105]
        input (1.0, 1.0) yields output [ 0.86855528]
        input (0.0, 1.0) yields output [-0.87403102]


    """