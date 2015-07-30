#!/ust/bin/python

import numpy as np
from module import * 

class CELayer(Module): 
    """"""

    def __init__(self, inputDim):
        """"""
        super(CELayer, self).__init__()
        self.inputDim = inputDim

    def forward(self, Xin, targetClass):
        """Forward the input vector Xin through the layer.

            INPUTS:
            -Xin: Input score vector of size inputDim
            -targetClass (integer): Expected class for Xin.

            OUTPUTS:
            -output (float): loss value for Xin.
                output = CE(Xin, targetClass) 
                       = -log(softmax(Xin))[targetClass]   The loss is the -log(softmax(Xin)) value for the expected class 
                       = -Xin_{targetClass} + log(Sum_i{exp(Xin_i)})

        """
        self.output = -Xin[targetClass] + np.log(np.sum(np.exp(Xin)))
        return self.output

    def backward(self, Xin, targetClass):
        """Used to propagate backward the derivatives in the network, compute the gradient with respect
           to the layer'sinput:
            - Xin: the input of the MSE layer.
           Output:
            - gradInput: d(Error)/d(Xin)
        """
        self.gradInput = np.exp(Xin)
        self.gradInput /= np.sum(self.gradInput)
        self.gradInput[targetClass] -= 1.
        return self.gradInput

    def get_probabilities(self, Xin):
        """"""
        prob = np.exp(Xin)
        prob /= np.sum(prob)
        return prob 

    def jacobian_check(self, eps=1e-5):
        """""" 
        for _ in xrange(100):
            currentInput = 0.1 * np.random.randn(self.inputDim)
            currentClass = np.random.randint(self.inputDim)
            gradInput = self.backward(currentInput, currentClass)
            estimatedGradInput = np.zeros(self.inputDim)
            #We need to compute the gradient wrt. all the input dimensions
            for i in xrange(self.inputDim):
                currentInput[i] += eps
                outputPlus = self.forward(currentInput, currentClass)
                currentInput[i] -= 2*eps
                outputMinus = self.forward(currentInput, currentClass)
                currentInput[i] += eps
                estimatedGradInput[i] = (outputPlus - outputMinus) / (2*eps)
            similarity = relative_error(estimatedGradInput, gradInput)
            print "Checking the jacobian matrix: ","ok" if similarity < eps else "error"

    def copy(self, shareWeights):
        """Return a new instance with similar parameters."""
        return CELayer(self.inputDim)

def relative_error(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)/np.linalg.norm(vec1 + vec2)
    
if __name__ == "__main__":
    """"""

    ce = CELayer(100)
    ce.jacobian_check()
