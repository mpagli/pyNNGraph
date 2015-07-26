#!/usr/bin/python

import numpy as np

class Module(object):
    """"""

    def __init__(self):
        """Constructor:
            -gradInput: gradient of the Error wrt. the previous module.

                        Err
                         |
                         |
                       out_2
                       MODULE
                         |
                         |
                       out_1
                       MODULE
                         |
                         |
                       in_1

            For the first module: 

                gradInput = d(Err)/d(in_1) = d(Err)/d(out_2) * d(out_2)/d(out_1) * d(out_1)/d(in_1)

            During backpropagation, the gradInput is will be updated and pass through each module. 

            Its value can then be used to compute the gradient wrt. the parameters of the modules:

                gradParam = d(Err)/d(param) = d(Err)/d(out_2) * d(out_2)/d(out_1) * d(out_1)/d(param)
                                             |-------- previous gradInput -------|

            -output: the output of the module.

        """
        self.gradInput = np.array([])
        self.output = np.array([])
        self.receiveGradFrom = []   #used only with network.py
        self.receiveInputFrom = []  #used only with network.py

    def parameters(self):
        """"""
        raise NotImplementedError("Subclass must implement abstract method")

    def forward(self):
        """"""
        raise NotImplementedError("Subclass must implement abstract method")

    def backward(self):
        """"""
        raise NotImplementedError("Subclass must implement abstract method")

    def jacobian_check(self):
        """"""
        raise NotImplementedError("Subclass must implement abstract method")

    def reset_grad_param(self):
        """"""
        raise NotImplementedError("Subclass must implement abstract method")