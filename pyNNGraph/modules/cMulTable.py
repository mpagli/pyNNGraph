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
        self.output = np.multiply(Xins)
        return self.output

    def backward(self, Xins, gradOutput):
        """Used to propagate backward the derivatives in the network, compute the gradient with respect
           to the module's input (gradInput):
            - gradOutput: the derivative backpropagated d(Error)/d(output)
            - Xins: input vectors
           Output:
            - gradInput: d(Error)/d(output) * jacobian = d(Error)/d(Xin) = gradOutput for all paths.
        """
        #Sorry for the list comprehension:
        self.gradInput = [ np.multiply(gradOutput, np.multiply(np.array(\
                         [x for j,x in enumerate(Xins) if i != j]\
                         ))) for i in xrange(len(Xins)) ] 
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
    pass

        

