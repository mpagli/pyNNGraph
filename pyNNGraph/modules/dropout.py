#!/ust/bin/python

import numpy as np 
from module import *

class Dropout(Module): 
    """"""

    def __init__(self, inputDim, p=0.5):
        """"""
        super(Dropout, self).__init__()
        self.p = 1. - p #Probability of a value to be kept.
        self.inputDim = inputDim
        self.mask = np.array([])
        self.train = True #If true then dropout is applied.

    def forward(self, Xin):
        """Forward the input vector Xin through the layer: output = mask * Xin where '*' is the 
           element-wise multiplication and mask is a vector of bernoulli samples."""
        if self.train:
            self.mask = np.random.binomial(1, self.p, self.inputDim)
            self.output = np.multiply(Xin, self.mask)
        else:
            self.output = np.multiply(Xin, self.p) #output is scaled when dropout is OFF.
        return self.output

    def backward(self, Xin, gradOutput):
        """Used to propagate backward the derivatives in the network, compute the gradient with respect
           to the module's input (gradInput):
            - gradOutput: the derivative backpropagated d(Error)/d(output)
            - Xin: input vector
           Output:
            - gradInput: d(Error)/d(output) * jacobian = d(Error)/d(Xin)
        """
        if self.train:
            self.gradInput = np.multiply(gradOutput, self.mask)
        else:
            self.gradInput = np.multiply(gradOutput, self.p)
        return self.gradInput

    def reset_grad_param(self):
        """"""
        return

    def parameters(self):
        """"""
        return

    def reset_grad_param(self):
        """"""
        return

    def compute_jacobian(self, Xin):
        """"""
        return

    def jacobian_check(self, eps=1e-5):
        """"""
        return

    def copy(self, shareWeights):
        """Return a new instance with similar parameters."""
        newNode = Dropout(self.inputDim, self.p)
        #newNode.receiveGradFrom = self.receiveGradFrom[:]
        #newNode.receiveInputFrom = self.receiveInputFrom[:]
        return newNode

if __name__ == "__main__":
    pass

        

