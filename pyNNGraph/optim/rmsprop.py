#!/usr/bin/python

import numpy as np

def RMSprop(feval, netParams, optimConf, optimState):
    """Perform one step of rmsprop optimization. 

        -feval: a function evaluating a network with parameters 'netParams',
         the prototype is feval(params). The input on which the network is
         evaluated is handled inside feval.

        -netParams: an array containing a reference to the network parameters.

        -optimConf: a hash table containing the parameters of the optimizer.

            optimConf.learningRate
            optimConf.decayRate

        -optimState: a hash table keeping track of the variables required by
         the optimizer (e.g. the previous gradient if we want to use momentum).

    """
    lr = optimConf['learningRate'] if 'learningRate' in optimConf else 1e-2
    dr = optimConf['decayRate'] if 'decayRate' in optimConf else 0.99
    
    eps = 1e-8

    #evaluate the network for the given parameters, and computing the derivatives
    loss, dfdp = feval(netParams)

    #update the moving average
    if 'm' not in optimState:
        optimState['m'] = [np.zeros(x.shape) for x in dfdp]
    for idx in xrange(len(optimState['m'])): 
        optimState['m'][idx] *= dr
        optimState['m'][idx] += np.multiply((1. - dr), np.multiply(dfdp[idx], dfdp[idx]))

    #parameters update
    for idx in xrange(len(netParams)):
        netParams[idx] += np.divide(np.multiply(-lr, dfdp[idx]), np.sqrt(optimState['m'][idx] + eps))

    return loss


