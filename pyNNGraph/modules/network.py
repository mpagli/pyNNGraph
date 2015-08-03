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
        for alias, node in self.nodesTable.items():
            node.alias = alias
        self.inputNodes = inputNodes
        self.outputNodes = outputNodes
        self.evaluationSequence = evaluationSequence
        self.recConnexions = [] #stores tuples representing the recurrent connexions 
        self.fwdConnexions = [] #stores tuples representing the forward connexions

    def link_nodes(self, sourceNode, targetNode):
        """Create an oriented connexion between two nodes. The source node send the signal 
        forward, the target node send the derivative backward.
        """
        self.fwdConnexions.append((sourceNode,targetNode))
        self.nodesTable[sourceNode].receiveGradFrom.append(targetNode)
        self.nodesTable[targetNode].receiveInputFrom.append(sourceNode)

    def recurrent_connexion(self, sourceNode, targetNode):
        """Create a connexion between the network and its copy at time t+1. Only effective
        after a call to unwrap 
        """
        self.recConnexions.append((sourceNode, targetNode))

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
            T[moduleName].push_forward(T)
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
            T[moduleName].push_backward(T)
        #Finally we need to propagate into the input nodes:
        for idx, inputName in enumerate(self.inputNodes):
            T[inputName].push_backward(T, Xins[idx])

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

    def _save_copy_for_prediction(self):
        """For the copy:
            inputNodes = self.inputNodes + [recInput1, recInput2, ...] -> recInput1 = h1_{t-1}
            outputNodes = self.outputNodes + [recOutput1, recOutput2, ...] -> recOutput1 = h1_{t} (saved for next timestep)
        """
        nodesTable = self.nodesTable.copy() #shallow copy of nodesTable, so we are reusing the same nodes
        inputNodes = self.inputNodes[:]
        outputNodes = self.outputNodes[:]
        evaluationSequence = self.evaluationSequence[:]
        self.recInputs = [] #contains the varying recurrent inputs 
        for sourceName, targetName in self.recConnexions:
            outputNodes.append(sourceName)
            inputNodes.append(targetName)
            evaluationSequence.remove(targetName)
            inputDim = nodesTable[targetName].inputDim
            self.recInputs.append(np.zeros(inputDim))
        self.rnnCopy = Network(nodesTable, inputNodes, outputNodes, evaluationSequence)

    def sequence_prediction(self, Xins):
        """Use self.rnnCopy to feed an input while saving the hidden states. We can feed a sequence
           element by element and get a prediction
        """
        outs = self.rnnCopy.forward(Xins+self.recInputs)
        self.recInputs = outs[-len(self.recConnexions):]
        return outs[:-len(self.recConnexions)]

    def reset_recurrent_states(self):
        """"""
        for array in self.recInputs:
            array.fill(0.)

    def unwrap(self, seqLength):
        """
        """
        self._save_copy_for_prediction() #save a copy of the net to simply get predictions after learning 
        if len(self.recConnexions) == 0:
            return 
        #Add all the missing nodes of the unwrapped net.
        for nodeName in self.nodesTable.keys():
            for t in xrange(1,seqLength):
                prefix = '_t'+str(t)
                newNodeName = nodeName + prefix
                newNode = self.nodesTable[nodeName].copy(shareWeights=True)
                self.nodesTable[newNodeName] = newNode
                self.nodesTable[newNodeName].alias = newNodeName
        #Add all the forward connexions 
        for sourceNode, targetNode in self.fwdConnexions[:]:
            for t in xrange(1,seqLength):
                prefix = '_t'+str(t)
                self.link_nodes(sourceNode+prefix, targetNode+prefix)
        #Add the new input nodes to self.inputNodes
        for inputNode in self.inputNodes[:]:
            for t in xrange(1,seqLength):
                prefix = '_t'+str(t)
                self.inputNodes.append(inputNode+prefix)
        #Add the new output nodes to self.outputNodes
        for outputNode in self.outputNodes[:]:
            for t in xrange(1,seqLength):
                prefix = '_t'+str(t)
                self.outputNodes.append(outputNode+prefix)
        #Add the recurrent connexions
        for sourceName, targetName in self.recConnexions:
            for t in xrange(0,seqLength-1):
                if t > 0:
                    prefixSource = '_t'+str(t)
                else:
                    prefixSource = ''
                prefixTarget = '_t'+str(t+1)
                self.link_nodes(sourceName+prefixSource, targetName+prefixTarget)
        #modify the evaluation sequence
        evalSeqCopy = self.evaluationSequence[:]
        for t in xrange(1,seqLength): 
            prefix = '_t'+str(t)
            for nodeName in evalSeqCopy:
                self.evaluationSequence.append(nodeName+prefix)


    def training_mode_ON(self):
        """When using dropout you need to set the train variable of the dropout module
           to True only when training. Moreover, if you try to check the gradient with
           train=True it will fail. 
        """
        for moduleName in self.nodesTable:
            if hasattr(self.nodesTable[moduleName], 'train'):
                self.nodesTable[moduleName].train = True

    def training_mode_OFF(self):
        """"""
        for moduleName in self.nodesTable:
            if hasattr(self.nodesTable[moduleName], 'train'):
                self.nodesTable[moduleName].train = False

    def copy(self, shareWeights=False):
        """Return a deep copy of the network"""
        nodesTable = {}
        for nodeName in self.nodesTable:
            nodesTable[nodeName] = self.nodesTable[nodeName].copy(shareWeights)
        inputNodes = self.inputNodes.copy()
        outputNodes = self.outputNodes.copy()
        evaluationSequence = self.evaluationSequence.copy()
        return Network(nodesTable, inputNodes, outputNodes, evaluationSequence)

    def gradient_checking(self, eps=1e-6):
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
        for _ in xrange(1):
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
