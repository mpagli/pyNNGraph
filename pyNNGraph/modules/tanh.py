#!/ust/bin/python

import numpy as np 
from module import *

class Tanh(Module): 
    """"""

    def __init__(self, inputDim):
        """"""
        super(Tanh, self).__init__()
        self.inputDim = inputDim
        self.jacobian = np.zeros(self.inputDim) #we store only the diagonal

    def forward(self, Xin):
        """Forward the input vector Xin through the layer: output = tanh(Xin) """
        self.output = np.tanh(Xin)
        return self.output

    def backward(self, Xin, gradOutput):
        """Used to propagate backward the derivatives in the network, compute the gradient with respect
           to the module's input (gradInput):
            - gradOutput: the derivative backpropagated d(Error)/d(output)
            - Xin: input vector
           Output:
            - gradInput: d(Error)/d(output) * jacobian = d(Error)/d(Xin)
        """
        #self.jacobian = (1 - np.power(np.tanh(Xin), 2)) #d(output)/d(input) = diag(self.jacobian)
        self.jacobian = (1 - np.power(self.output, 2)) #d(output)/d(input) = diag(self.jacobian)
        self.gradInput = np.multiply(gradOutput, self.jacobian)
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
        self.jacobian = self.jacobian = (1 - np.power(np.tanh(Xin), 2))
        return self.jacobian

    def jacobian_check(self, eps=1e-5):
        """Check if the jacobian matrix is correct.
        """
        for _ in xrange(100):
            currentInput = 0.1 * np.random.randn(self.inputDim)
            self.compute_jacobian(currentInput)
            estimatedJacobian = np.zeros((self.inputDim, self.inputDim))
            #We need to compute the gradient wrt. all the input dimensions
            for i in xrange(self.inputDim):
                currentInput[i] += eps
                outputPlus = np.tanh(currentInput)
                currentInput[i] -= 2*eps
                outputMinus = np.tanh(currentInput)
                currentInput[i] += eps
                estimatedJacobian[i,:] = np.divide((np.subtract(outputPlus, outputMinus)), 2*eps)
            similarity = relative_error(estimatedJacobian.reshape(self.inputDim * self.inputDim), np.diag(self.jacobian).reshape(self.inputDim * self.inputDim))
            print "Checking the jacobian matrix: ","ok" if similarity < eps else "error"

    def copy(self, shareWeights):
        """Return a new instance with similar parameters."""
        newNode = Tanh(self.inputDim)
        #newNode.receiveGradFrom = self.receiveGradFrom[:]
        #newNode.receiveInputFrom = self.receiveInputFrom[:]
        return newNode

def relative_error(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)/np.linalg.norm(vec1 + vec2)


if __name__ == "__main__":
    """Runing tests"""

    tanh = Tanh(30)
    tanh.jacobian_check()

        

