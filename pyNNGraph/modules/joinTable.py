#!/ust/bin/python

import numpy as np 
from module import *

class JoinTable(Module): 
    """"""

    def __init__(self, inputDim):
        """"""
        super(JoinTable, self).__init__()
        self.inputDim = inputDim
        self.gradInput = [] #we have several gradInputs, one for each input signal 

    def forward(self, Xins):
        """Forward the input vectors Xins through the layer: output = concatenate(Xins) """ 
        self.output = np.concatenate(Xins, axis=0)
        return self.output

    def backward(self, Xins, gradOutput):
        """Used to propagate backward the derivatives in the network, compute the gradient with respect
           to the module's input (gradInput):
            - gradOutput: the derivative backpropagated d(Error)/d(output)
            - Xins: input vectors
           Output:
            - gradInput: d(Error)/d(output) * jacobian = d(Error)/d(Xin) = gradOutput for all paths.
        """
        self.gradInput = [] 
        split = 0
        for Xin in Xins:
            self.gradInput.append(gradOutput[split:split+len(Xin)])
            split = len(Xin)
        return self.gradInput

    def parameters(self):
        """"""
        return

    def reset_grad_param(self):
        """"""
        return

    def jacobian_check(self, eps=1e-5):
        """"""
        return

    def copy(self, shareWeights):
        """Return a new instance with similar parameters."""
        newNode = JoinTable(self.inputDim)
        #newNode.receiveGradFrom = self.receiveGradFrom[:]
        #newNode.receiveInputFrom = self.receiveInputFrom[:]
        return newNode


if __name__ == "__main__":
    pass

        

