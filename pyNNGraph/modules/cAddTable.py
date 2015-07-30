#!/ust/bin/python

import numpy as np 
from module import *

class CAddTable(Module): 
    """"""

    def __init__(self, inputDim):
        """"""
        super(CAddTable, self).__init__()
        self.inputDim = inputDim
        self.gradInput = [] #we have several gradInputs, one for each input signal 
        #self.jacobian: since we are adding vectors, the jacobian is the identity for each path

    def forward(self, Xins):
        """Forward the input vectors Xins through the layer: output = sum(Xins) """ 
        self.output = np.sum(Xins, axis=0)
        return self.output

    def backward(self, Xins, gradOutput):
        """Used to propagate backward the derivatives in the network, compute the gradient with respect
           to the module's input (gradInput):
            - gradOutput: the derivative backpropagated d(Error)/d(output)
            - Xins: input vectors
           Output:
            - gradInput: d(Error)/d(output) * jacobian = d(Error)/d(Xin) = gradOutput for all paths.
        """
        self.gradInput = gradOutput#[gradOutput for _ in Xins]
        return self.gradInput

    def parameters(self):
        """"""
        return

    def reset_grad_param(self):
        """"""
        return

    def jacobian_check(self, eps=1e-5):
        """The jacobian is the identity. For two input vector x, y:

                out1 = x1+y1
                out2 = x2+y2

                 | d(out1)/d(x1) d(out1)/d(x2) |   | 1  0 |
            Jx = | d(out2)/d(x1) d(out2)/d(x2) | = | 0  1 | , idem for Jy.
        """
        return

    def copy(self, shareWeights):
        """Return a new instance with similar parameters."""
        newNode = CAddTable(self.inputDim)
        #newNode.receiveGradFrom = self.receiveGradFrom[:]
        #newNode.receiveInputFrom = self.receiveInputFrom[:]
        return newNode


if __name__ == "__main__":
    pass

        

