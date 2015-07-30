#!/ust/bin/python

import numpy as np
from module import * 

class MSELayer(Module): 
    """"""

    def __init__(self, inputDim):
        """"""
        super(MSELayer, self).__init__()
        self.inputDim = inputDim

    def forward(self, Xin, targets):
        """Forward the input vector Xin through the layer: output = MSE(Xin, targets) """
        self.output = np.power(np.subtract(Xin, targets) ,2)
        return self.output

    def backward(self, Xin, targets):
        """Used to propagate backward the derivatives in the network, compute the gradient with respect
           to the layer'sinput:
            - Xin: the input of the MSE layer.
           Output:
            - gradInput: d(Error)/d(Xin)
        """
        self.gradInput = np.multiply(2.0, np.subtract(Xin, targets))
        return self.gradInput

    def jacobian_check(self, eps=1e-5):
        """Check if the jacobian matrix is correct. For this layer since the jacobian matrix is diagonal
           and this layer being always at the end of a network, the jacobian matrix is not stored. It 
           should be:
                           | 2*(x1-t1)       0            0     |
                     J =   |     0       2*(x2-t2)        0     |       For a layer of size 3.
                           |     0           0        2*(x3-t3) |
                           
            diag(J) = gradInput = [1, 1, 1] * J
        """ 
        for _ in xrange(100):
            currentInput = 0.1 * np.random.randn(self.inputDim)
            currentTarget = 0.1 * np.random.randn(self.inputDim)
            jacobian = self.backward(currentInput, currentTarget)
            computedJacobian = np.zeros((self.inputDim, self.inputDim))
            #We need to compute the gradient wrt. all the input dimensions
            for i in xrange(self.inputDim):
                currentInput[i] += eps
                outputPlus = self.forward(currentInput, currentTarget)
                currentInput[i] -= 2*eps
                outputMinus = self.forward(currentInput, currentTarget)
                currentInput[i] += eps
                computedJacobian[i,:] = np.divide((np.subtract(outputPlus, outputMinus)), 2*eps)
            similarity = relative_error(computedJacobian.reshape(self.inputDim * self.inputDim), np.diag(jacobian).reshape(self.inputDim * self.inputDim))
            print "Checking the jacobian matrix: ","ok" if similarity < eps else "error"

    def copy(self, shareWeights):
        """Return a new instance with similar parameters."""
        return MSELayer(self.inputDim)

def relative_error(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)/np.linalg.norm(vec1 + vec2)


if __name__ == "__main__":
    """Runing tests"""

    mse = MSELayer(30)
    mse.jacobian_check()