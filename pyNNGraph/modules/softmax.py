#!/ust/bin/python

import numpy as np 
from module import *

class Softmax(Module): 
    """"""

    def __init__(self, inputDim):
        """"""
        super(Softmax, self).__init__()
        self.inputDim = inputDim
        self.jacobian = np.zeros((self.inputDim,self.inputDim))

    def forward(self, Xin):
        """Forward the input vector Xin through the layer: output = Softmax(Xin) """
        self.output = np.exp(Xin)
        self.output /= np.sum(self.output)
        return self.output

    def backward(self, Xin, gradOutput):
        """Used to propagate backward the derivatives in the network, compute the gradient with respect
           to the module's input (gradInput):
            - gradOutput: the derivative backpropagated d(Error)/d(output)
            - Xin: input vector
           Output:
            - gradInput: d(Error)/d(output) * jacobian = d(Error)/d(Xin)
        """
        self.jacobian = np.diag(self.output)
        self.jacobian -= np.outer(self.output, self.output)
        self.gradInput = np.dot(gradOutput, self.jacobian)
        return self.gradInput

    def reset_grad_param(self):
        """"""
        return

    def parameters(self):
        """"""
        return

    def compute_jacobian(self, Xin):
        """"""
        output = np.exp(Xin)
        output /= np.sum(output) 
        self.jacobian = np.diag(output)
        self.jacobian -= np.outer(output, output)
        return self.jacobian

    def jacobian_check(self, eps=1e-5):
        """Check if the jacobian matrix is correct:

                    | h1.(1-h1)    -h1.h2      -h1.h3   |
                J = |  -h1.h2     h2.(1-h2)    -h3.h2   |     h1 = exp(x1)/sum_i(exp(x1))
                    |  -h1.h3      -h2.h3     h3.(1-h3) |

        """
        for _ in xrange(100):
            currentInput = 0.1 * np.random.randn(self.inputDim)
            self.compute_jacobian(currentInput)
            estimatedJacobian = np.zeros((self.inputDim, self.inputDim))
            #We need to compute the gradient wrt. all the input dimensions
            for i in xrange(self.inputDim):
                currentInput[i] += eps
                outputPlus = self.forward(currentInput)
                currentInput[i] -= 2*eps
                outputMinus = self.forward(currentInput)
                currentInput[i] += eps
                estimatedJacobian[i,:] = np.divide((np.subtract(outputPlus, outputMinus)), 2*eps)
            similarity = relative_error(estimatedJacobian.reshape(self.inputDim * self.inputDim), self.jacobian.reshape(self.inputDim * self.inputDim))
            print "Checking the jacobian matrix: ","ok" if similarity < eps else "error"

def relative_error(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)/np.linalg.norm(vec1 + vec2)


if __name__ == "__main__":
    """Runing tests"""

    module = Softmax(3)

    out = module.forward(np.array([1.,2.,4.]))
    print "output: ",out, sum(list(out))

    gradInput = module.backward(np.array([1.,1.,1.]) ,np.array([1.,1.,1.]))
    print "gradInput: ", gradInput

    print module.jacobian

    module.jacobian_check()
