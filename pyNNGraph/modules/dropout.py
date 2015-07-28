#!/ust/bin/python

import numpy as np 
from module import *

class Dropout(Module): 
    """"""

    def __init__(self, inputDim, p=0.5):
        """"""
        super(Dropout, self).__init__()
        self.p = 1. - p
        self.inputDim = inputDim
        self.mask = np.array([])
        self.train = True

    def forward(self, Xin):
        """Forward the input vector Xin through the layer: output = tanh(Xin) """
        if self.train:
            self.mask = np.random.binomial(1, self.p, self.inputDim)
            self.output = np.multiply(Xin, self.mask)
        else:
            self.output = np.multiply(Xin, self.p)
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

if __name__ == "__main__":
    pass

        

