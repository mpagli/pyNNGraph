#!/ust/bin/python

import numpy as np 
from module import *

class CMulTable(Module): 
    """"""

    def __init__(self, inputDim):
        """"""
        super(CMulTable, self).__init__()
        self.inputDim = inputDim
        self.gradInput = [] #we have several gradInputs, one for each input signal 
        #self.jacobian

    def forward(self, Xins):
        """Forward the input vectors Xins through the layer: output = multiply(Xins) """ 
        self.output = Xins[0].copy()
        for idx in xrange(1,len(Xins)):
            self.output *= Xins[idx]
        return self.output

    def backward(self, Xins, gradOutput):
        """Used to propagate backward the derivatives in the network, compute the gradient with respect
           to the module's input (gradInput):
            - gradOutput: the derivative backpropagated d(Error)/d(output)
            - Xins: input vectors
           Output:
            - gradInput: d(Error)/d(output) * jacobian .
        """
        if self.gradInput == []:
            self.gradInput = [None]*len(Xins)
        for i in xrange(len(Xins)):
            self.gradInput[i] = gradOutput.copy()
            for j in xrange(len(Xins)):
                if i != j:
                    self.gradInput[i] *= Xins[j]
        return self.gradInput

    def parameters(self):
        """"""
        return

    def reset_grad_param(self):
        """"""
        return

    def jacobian_check(self, eps=1e-5):
        """ Computing the jacobian: 

            for 3 input vectors x,y,z. out = x*y*z (* being the element-wize multiplication)

            In the case x,y and z are of dimension 3:

                 | y1*z1     0       0   |           | x1*z1     0       0   | 
            Jx = |   0     y2*z2     0   |      Jy = |   0     x2*z2     0   |
                 |   0       0     y3*z3 |           |   0       0     x3*z3 |


                 | y1*x1     0       0   |
            Jz = |   0     y2*x2     0   |
                 |   0       0     y3*x3 |

        """
        return

    def copy(self, shareWeights):
        """Return a new instance with similar parameters."""
        newNode = CMulTable(self.inputDim)
        #newNode.receiveGradFrom = self.receiveGradFrom[:]
        #newNode.receiveInputFrom = self.receiveInputFrom[:]
        return newNode


if __name__ == "__main__":
    
    m = CMulTable(3)

    Xins = [np.array([1.,1.,2.]), np.array([1.,2.,3.])]

    print m.forward(Xins)

    print m.backward(Xins, np.array([2.,2.,2.]))

        

