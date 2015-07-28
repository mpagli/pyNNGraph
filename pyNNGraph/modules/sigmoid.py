#!/ust/bin/python

import numpy as np 
from module import *

class Sigmoid(Module): 
    """"""

    def __init__(self, inputDim):
        """"""
        super(Sigmoid, self).__init__()
        self.inputDim = inputDim
        self.jacobian = np.zeros(self.inputDim) #we store only the diagonal
        self.output = np.zeros(self.inputDim)

    def forward(self, Xin):
        """Forward the input vector Xin through the layer: output = tanh(Xin) """
        self.output.fill(1.) 
        self.output /= 1. + np.exp(-Xin)
        return self.output

    def backward(self, Xin, gradOutput):
        """Used to propagate backward the derivatives in the network, compute the gradient with respect
           to the module's input (gradInput):
            - gradOutput: the derivative backpropagated d(Error)/d(output)
            - Xin: input vector
           Output:
            - gradInput: d(Error)/d(output) * jacobian = d(Error)/d(Xin)
        """
        self.jacobian = np.multiply(self.output, np.add(1., -self.output))
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
        output = np.ones(self.inputDim)
        output = np.divide(output ,np.add(1., np.exp(-Xin)))
        self.jacobian = np.multiply(output, np.add(1., -output))
        return self.jacobian

    def jacobian_check(self, eps=1e-5):
        """Check if the jacobian matrix is correct.   
        """
        for _ in xrange(100):
            currentInput = 0.1 * np.random.randn(self.inputDim)
            self.compute_jacobian(currentInput)
            #print self.jacobian
            estimatedJacobian = np.zeros((self.inputDim, self.inputDim))
            #We need to compute the gradient wrt. all the input dimensions
            for i in xrange(self.inputDim):
                currentInput[i] += eps
                outputPlus = self.forward(currentInput).copy()
                currentInput[i] -= 2*eps
                outputMinus = self.forward(currentInput).copy()
                currentInput[i] += eps
                estimatedJacobian[i,:] = np.divide((np.subtract(outputPlus, outputMinus)), 2*eps)
            similarity = relative_error(estimatedJacobian.reshape(self.inputDim * self.inputDim), np.diag(self.jacobian).reshape(self.inputDim * self.inputDim))
            print "Checking the jacobian matrix: ","ok" if similarity < eps else "error"

def relative_error(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)/np.linalg.norm(vec1 + vec2)


if __name__ == "__main__":
    """Runing tests"""

    sigmoid = Sigmoid(3)
    sigmoid.jacobian_check()

        

