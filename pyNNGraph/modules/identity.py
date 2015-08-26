#!/ust/bin/python

import numpy as np 
from module import *

class Identity(Module): 
    """"""

    def __init__(self, inputDim):
        """"""
        super(Identity, self).__init__()
        self.inputDim = inputDim

    def forward(self, Xin):
        """"""
        self.output = Xin
        return self.output

    def backward(self, Xin, gradOutput):
        """"""
        self.gradInput = gradOutput
        return self.gradInput

    def reset_grad_param(self):
        """"""
        return

    def parameters(self):
        """"""
        return

    def compute_jacobian(self, Xin):
        """"""
        return

    def push_forward(self, nodesTable):
        """"""
        if len(self.receiveInputFrom) == 1:
            source = self.receiveInputFrom[0]
            Xin = nodesTable[source].get_output(self.alias)
            return self.forward(Xin)
        elif len(self.receiveInputFrom) == 0: #assume zeros vector as input (usefull for recurrent nets)
            return self.forward(np.zeros(self.inputDim))
        else:
            #throw error here
            return

    def push_backward(self, nodesTable, Xin=None):
        """"""
        if len(self.receiveGradFrom) == 1:
            target = self.receiveGradFrom[0]
            gradOutput = nodesTable[target].get_gradInput(self.alias)
            return self.backward(None, gradOutput)
        elif len(self.receiveGradFrom) == 0:
            #throw error here
            return
        else:   #we add al the gradOutpus
            gradOutput = np.zeros(self.inputDim)
            for nodeName in self.receiveGradFrom:
                gradOutput += nodesTable[nodeName].get_gradInput(self.alias)
            return self.backward(None, gradOutput)

    def get_gradInput(self, targetNode):
        """"""
        return self.gradInput

    def get_output(self, sourceNode):
        """"""
        return self.output

    def jacobian_check(self, eps=1e-5):
        """"""
        return

    def copy(self, shareWeights):
        """Return a new instance with similar parameters."""
        newNode = Identity(self.inputDim)
        #newNode.receiveGradFrom = self.receiveGradFrom[:]
        #ewNode.receiveInputFrom = self.receiveInputFrom[:]
        return newNode


if __name__ == "__main__":
    """Runing tests"""

