#!/usr/bin/python

import numpy as np
import MSELayer as mse

class Network(object):
    """"""

    def __init__(self, nodesTable, inputNodes, outputNodes, evaluationSequence):
        """Let's imagine we have the following network (two inputs, two outputs):

            X1 -----Linear(100,20)--Tanh(20)--|                                    |---- Y1
                                              |                                    |
                                        CAddTable(20)----Linear(20,5)----Tanh(5)---|
                                              |                                    |
            X2 -----Linear(50,20)---Tanh(20)--|                                    |----Linear(5,5)----Tanh(5)---- Y2

        Here is what the parameters would be:

            -nodesTable: a hash map containing all the nodes of the networks, the key is some name.

                        nodesTable = { 
                            'input1':   nn.Linear(100,20), 
                            'input2':   nn.Linear(50,20), 
                            'tanh1':    nn.Tanh(20), 
                            'tanh2':    nn.Tanh(20), 
                            'add':      nn.CAddTable(20),
                            'linear1':  nn.Linear(20.5),
                            'linear2':  nn.Linear(5,5),
                            'tanh3':    nn.Tanh(5),
                            'tanh4':    nn.Tanh(5)
                            }

            -inputNodes: a list containing the names of the input nodes. The order is important!

                        inputNodes = ['input1', 'input2']

            -outputNodes: a list containing the names of the output nodes. The order matters!

                        outputNodes = ['tanh3', 'tanh4']

            -evaluationSequence: the order we need to follow to perform the forward pass. No need 
             to put the input nodes in the sequence.

                        evaluationSequence = ['tanh1','tanh2','add','linear1','tanh3','linear2','tanh4']
        """
        self.nodesTable = nodesTable
        self.inputNodes = inputNodes
        self.outputNodes = outputNodes
        self.evaluationSequence = evaluationSequence

    def link_nodes(self, sourceNode, targetNode):
        """Create an oriented connexion between two nodes. The source node send the signal 
        forward, the target node send the derivative backward.
        """
        self.nodesTable[sourceNode].receiveGradFrom.append(targetNode)
        self.nodesTable[targetNode].receiveInputFrom.append(sourceNode)

    def forward(self, Xins):
        """Forwards the inputs through the network. The Error is not computed here,
           it is computed by external modules. 
            -Xins: a vector containing the input vectors. The order is the same
             as in self.inputNodes.
        """
        T = self.nodesTable
        #We start at the input nodes:
        for idx, inputName in enumerate(self.inputNodes):
            T[inputName].forward(Xins[idx]) #the output of each module is saved in module.output
        #We then proceed to forward the inputs in the entire network
        for moduleName in self.evaluationSequence:
            sources = T[moduleName].receiveInputFrom
            #A module can have several input module (e.g. CAddTable, CMulTable ...)
            if len(sources) > 1:
                T[moduleName].forward([T[source].output for source in sources])
            else:
                T[moduleName].forward(T[sources[0]].output)
        #Now we passed trough all the nodes, we return the outputs 
        outputVector = [None]*len(self.outputNodes)
        for idx, outputName in enumerate(self.outputNodes):
            outputVector[idx] = T[outputName].output
        return outputVector

    def backward(self, Xins, gradOutputs):
        """
            -Xins: vector of input vectors
            -gradOutputs: we have one gradOutput vector per output. The gradOutputs come
             from the error metrics. 
        """
        T = self.nodesTable
        #We start by the output nodes:
        for idx, outputName in enumerate(self.outputNodes):
            sources = T[outputName].receiveInputFrom
            if len(sources) > 1:
                T[outputName].backward([T[source].output for source in sources], gradOutputs[idx])
            else:
                T[outputName].backward(T[sources[0]].output, gradOutputs[idx])
        #We propagate the error backward in the network
        for moduleName in self.evaluationSequence[::-1]: #We go backward
            if moduleName in self.outputNodes:
                continue
            sources = T[moduleName].receiveInputFrom
            targets = T[moduleName].receiveGradFrom
            if len(targets) > 1:
                localGradOutputs = np.sum([T[target].gradInput for target in targets], axis=0)
            else:
                localGradOutputs = T[targets[0]].gradInput
            if len(sources) == 1:
                localXins = T[sources[0]].output
            else:
                localXins = [T[source].output for source in sources]
            T[moduleName].backward(localXins, localGradOutputs)
        #Finally we need to propagate into the input nodes:
        for idx, inputName in enumerate(self.inputNodes):
            targets = T[inputName].receiveGradFrom
            if len(targets) > 1:
                localGradOutputs = np.sum([T[target].gradInput for target in targets], axis=0)
            else:
                localGradOutputs = T[targets[0]].gradInput
            T[inputName].backward(Xins[idx], localGradOutputs)

    def get_link_to_parameters(self):
        """Return a list of references to the weights and associated gradients. Carefull not
           to break the references (no realloc, use inplace operators)."""
        parametersList = []
        gradParametersList = []
        for moduleName in self.nodesTable:
            if hasattr(self.nodesTable[moduleName], 'weight'):
                parametersList.append(self.nodesTable[moduleName].weight)
                gradParametersList.append(self.nodesTable[moduleName].gradWeight)
            if hasattr(self.nodesTable[moduleName], 'bias'):
                parametersList.append(self.nodesTable[moduleName].bias)
                gradParametersList.append(self.nodesTable[moduleName].gradBias)
        return parametersList, gradParametersList

    def reset_grad_param(self):
        """Reset all the gradParam to 0. Necessary since the gradient is accumulated."""
        for moduleName in self.nodesTable:
            self.nodesTable[moduleName].reset_grad_param()

    def gradient_checking(self, eps=1e-5):
        """"""
        def get_goutputs(Xouts, ts, errN):
            gOutputs = []
            for idx, out in enumerate(Xouts):
                errN[idx].forward(out,ts[idx])
                gOutputs.append(errN[idx].backward(out,ts[idx]))
            return gOutputs

        def get_error(Xouts, ts, errN):
            errors = []
            for idx, out in enumerate(Xouts):
                errors.append(np.sum(errN[idx].forward(out,ts[idx])))
            return sum(errors)

        def relative_error(vec1, vec2):
            if len(vec1.shape) > 1:
                vec1 = vec1.reshape(vec1.shape[0]*vec1.shape[1],1)
                vec2 = vec2.reshape(vec2.shape[0]*vec2.shape[1],1)
                return np.linalg.norm(vec1 - vec2)/np.linalg.norm(vec1 + vec2)
            else:
                return np.linalg.norm(vec1 - vec2)/np.linalg.norm(vec1 + vec2)

        parametersList, gradParametersList = self.get_link_to_parameters()
        T = self.nodesTable
        relativeErrList = []
        for _ in xrange(20):
            Xins = []
            targets = []
            self.reset_grad_param()
            #generate fake inputs:
            for inputName in self.inputNodes:
                Xins.append(np.random.randn(T[inputName].inputDim))
            #generate fake targets:
            errorNodes = []
            for outputName in self.outputNodes:
                if hasattr(T[outputName], 'outputDim'):
                    targets.append(np.random.randn(T[outputName].outputDim))
                    errorNodes.append(mse.MSELayer(T[outputName].outputDim))
                else:
                    targets.append(np.random.randn(T[outputName].inputDim))
                    errorNodes.append(mse.MSELayer(T[outputName].inputDim))
            #compute the gradient the standard way:
            outs = self.forward(Xins)
            gradOutputs = get_goutputs(outs, targets, errorNodes)
            self.backward(Xins, gradOutputs)
            #modify one weight and propagate forward:
            approxGradParams = [x.copy() for x in gradParametersList]
            for idx in xrange(len(parametersList)):
                params = parametersList[idx]
                if len(params.shape) == 1:
                    for i in xrange(params.shape[0]):
                        params[i] += eps
                        outs = self.forward(Xins)
                        errorPlus = get_error(outs, targets, errorNodes)
                        params[i] -= 2.*eps
                        outs = self.forward(Xins)
                        errorMinus = get_error(outs, targets, errorNodes)
                        params[i] += eps
                        print (errorPlus - errorMinus)/(2*eps)
                        approxGradParams[idx][i] = (errorPlus - errorMinus)/(2*eps)
                else:
                    for i in xrange(params.shape[0]):
                        for j in xrange(params.shape[1]):
                            params[i] += eps
                            outs = self.forward(Xins)
                            errorPlus = get_error(outs, targets, errorNodes)
                            params[i][j] -= 2.*eps
                            outs = self.forward(Xins)
                            errorMinus = get_error(outs, targets, errorNodes)
                            params[i][j] += eps
                            approxGradParams[idx][i][j] = (errorPlus - errorMinus)/(2*eps)
                relativeErrList.append(relative_error(approxGradParams[idx],gradParametersList[idx]))
        return relativeErrList


if __name__ == '__main__':
    """"""
