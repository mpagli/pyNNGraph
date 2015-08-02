#!/ust/bin/python

import numpy as np 
from module import *

class SplitTable(Module): 
    """"""

    def __init__(self, inputDim, splitList):
        """"""
        super(SplitTable, self).__init__()
        self.inputDim = inputDim
        self.splitList = splitList

    def forward(self, Xin):
        """Forward the input vectors Xins through the layer: output = split(Xins, splitList) """ 
        self.output = [None]*len(self.splitList)
        split = 0
        for idx,splitSize in enumerate(self.splitList):
            self.output[idx] = Xin[split:split+splitSize]
            split = splitSize
        return self.output

    def backward(self, Xin, gradOutputs):
        """
        """
        self.gradInput = np.concatenate(gradOutputs)
        return self.gradInput

    def parameters(self):
        """"""
        return

    def reset_grad_param(self):
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

    def push_backward(self, nodesTable, Xins=[]):
        """"""
        gradOutputs = [None]*len(self.receiveGradFrom)
        for idx, nodeName in enumerate(self.receiveGradFrom):   
            gradOutputs[idx] = nodesTable[nodeName].get_gradInput(self.alias)
        return self.backward(None, gradOutputs)

    def get_gradInput(self, targetNode):
        """"""
        return self.gradInput

    def get_output(self, sourceNode):
        """"""
        idx = self.receiveGradFrom.index(sourceNode)
        return self.output[idx]

    def jacobian_check(self, eps=1e-5):
        """
        """
        return

    def copy(self, shareWeights):
        """Return a new instance with similar parameters."""
        newNode = SplitTable(self.inputDim, self.splitList[:])
        #newNode.receiveGradFrom = self.receiveGradFrom[:]
        #newNode.receiveInputFrom = self.receiveInputFrom[:]
        return newNode


if __name__ == "__main__":
    
    split = SplitTable(2*2, [2, 2])

    print split.forward(np.array([1,2,3,4]))

    print split.backward(None,split.forward(np.array([1,2,3,4])))

        

