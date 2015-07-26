#!/usr/bin/python

import numpy as np

def SGD(feval, netParams, optimConf, optimState):
    """Perform one step of stochastic gradient descent. 

        -feval: a function evaluating a network with parameters 'netParams',
         the prototype is feval(params). The input on which the network is
         evaluated is handled inside feval.

        -netParams: an array containing a reference to the network parameters.

        -optimConf: a hash table containing the parameters of the optimizer.

            optimConf.learningRate
            optimConf.learningRateDecay
            optimConf.weightDecay
            optimConf.momentum

        -optimState: a hash table keeping track of the variables required by
         the optimizer (e.g. the previous gradient if we want to use momentum).

    """
    lr = optimConf['learningRate'] if 'learningRate' in optimConf else 1e-3
    lrd = optimConf['learningRateDecay'] if 'learningRateDecay' in optimConf else 0
    wd = optimConf['weightDecay'] if 'weightDecay' in optimConf else 0 
    mom = optimConf['momentum'] if 'momentum' in optimConf else 0
    optimState['evalCounter'] = optimState['evalCounter'] if 'evalCounter' in optimState else 0

    #evaluate the network for the given parameters, and computing the derivatives
    loss, dfdp = feval(netParams)

    #apply weight decay: w = w*wd
    if wd != 0:
        for idx in xrange(len(dfdp)):
            netParams[idx] *= wd 

    #apply momentum: dfdx = dfdx + prevDfdx*mom
    if mom != 0:
        if 'dfdp' not in optimState: #no gradient saved
            optimState['dfdp'] = [x.copy() for x in dfdp]
        else:
            for idx in xrange(len(optimState['dfdp'])):
                optimState['dfdp'][idx] *= mom
                dfdp[idx] += optimState['dfdp'][idx]

    #learning rate decay: lr = lr / (1+nevals*lrd)
    currentLr = lr / (1+optimState['evalCounter'] * lrd)

    #parameters update
    for idx in xrange(len(netParams)):
        netParams[idx] += -currentLr * dfdp[idx]

    optimState['evalCounter'] += 1

    return loss


