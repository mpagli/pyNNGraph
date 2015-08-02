#!/usr/bin/python

import numpy as np
from module import *

class Linear(Module):
    """"""

    def __init__(self, inputDim, outputDim, stdv=None):
        """"""
        super(Linear, self).__init__()
        if not stdv :
            self.stdv = 3./np.sqrt(inputDim)
        else:
            self.stdv = stdv
        self.inputDim = inputDim
        self.outputDim = outputDim
        #self.layerJacobian = np.ones((self.outputDim, self.inputDim)) #Since it's a linear layer the jacobian is the transposed of the weights matrix
        self.weight = np.random.uniform(-self.stdv, self.stdv, (inputDim, outputDim))
        self.bias = np.random.uniform(-self.stdv, self.stdv, outputDim)
        self.gradWeight = np.zeros((inputDim, outputDim))
        self.gradBias = np.zeros(outputDim)

    def forward(self, Xin):
        """Forward the input vector Xin through the layer: output = W * Xin + bias"""
        self.output = np.add(np.dot(Xin, self.weight), self.bias)
        return self.output

    def backward(self, Xin, gradOutput):
        """Used to propagate backward the derivatives in the network, compute the gradient with respect
           to the module's input (gradInput) and parameters (gradWeight and gradBias):
            - gradOutput: the derivative backpropagated d(Error)/d(output)
            - Xin: input vector
           Output:
            - gradInput: d(Error)/d(output) * layerJacobian = d(Error)/d(Xin)
            - gradWeight: d(Error)/d(output) * d(output)/d(weights)
            - gradBias: d(Error)/d(output) * d(output)/d(bias)
        """
        self.gradInput = np.dot(gradOutput, self.weight.T)
        self.gradWeight += np.outer(Xin, gradOutput)#the gradients wrt. the params are accumulated
        self.gradBias += gradOutput
        return self.gradInput

    def reset_grad_param(self):
        """"""
        self.gradWeight.fill(0.)
        self.gradBias.fill(0.)

    def parameters(self):
        """"""
        return [(self.weight, self.bias), (self.gradWeight, self.gradBias)]

    def push_forward(self, nodesTable):
        """"""
        if len(self.receiveInputFrom) == 1:
            source = self.receiveInputFrom[0]
            Xin = nodesTable[source].get_output(self.alias)
            return self.forward(Xin)
        elif len(self.receiveInputFrom) == 0: #assume zeros vector as input (usefull for recurrent nets)
            return self.forward(np.zeros(self.inputDim))
        else:
            #throw error here
            return

    def push_backward(self, nodesTable, Xin=[]):
        """"""
        if Xin == []:
            if len(self.receiveInputFrom) == 0: 
                Xin = np.zeros(self.inputDim)
            else:
                source = self.receiveInputFrom[0]
                Xin = nodesTable[source].get_output(self.alias)
        if len(self.receiveGradFrom) == 1:
            target = self.receiveGradFrom[0]
            gradOutput = nodesTable[target].get_gradInput(self.alias)
            return self.backward(Xin, gradOutput)
        elif len(self.receiveGradFrom) == 0:
            #throw error here
            return
        else:   #we add al the gradOutpus
            gradOutput = np.zeros(self.inputDim)
            for nodeName in self.receiveGradFrom:
                gradOutput += nodesTable[nodeName].get_gradInput(self.alias)
            return self.backward(Xin, gradOutput)

    def get_gradInput(self, targetNode):
        """"""
        return self.gradInput

    def get_output(self, sourceNode):
        """"""
        return self.output

    def jacobian_check(self, eps=1e-5):
        """Check if the jacobian matrix is correct. For a linear layer the jacobian matrix is
           the transposed of the weights matrix.
        """
        for _ in xrange(100):
            currentInput = 0.1 * np.random.randn(self.inputDim)
            computedJacobian = np.zeros((self.inputDim, self.outputDim))
            #We need to compute the gradient wrt. all the input dimensions
            for i in xrange(self.inputDim):
                currentInput[i] += eps
                outputPlus = np.add(np.dot(self.weight.T, currentInput), self.bias)
                currentInput[i] -= 2*eps
                outputMinus = np.add(np.dot(self.weight.T, currentInput), self.bias)
                currentInput[i] += eps
                computedJacobian[i,:] = np.divide((np.subtract(outputPlus, outputMinus)), 2*eps)
            similarity = relative_error(computedJacobian.reshape(self.inputDim * self.outputDim), self.weight.reshape(self.inputDim * self.outputDim))
            print "Checking the jacobian matrix: ","ok" if similarity < eps else "error"

    def copy(self, shareWeights):
        """Return a new instance with similar parameters."""
        newNode = Linear(self.inputDim, self.outputDim, self.stdv)
        #newNode.receiveGradFrom = self.receiveGradFrom[:]
        #newNode.receiveInputFrom = self.receiveInputFrom[:]
        if shareWeights:
            newNode.weight = self.weight
            newNode.gradWeight = self.gradWeight
            newNode.bias = self.bias
            newNode.gradBias = self.gradBias
        return newNode

def relative_error(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)/np.linalg.norm(vec1 + vec2)


if __name__ == "__main__":
    """Runing tests"""

    lin = Linear(30,10)
    lin.jacobian_check()


