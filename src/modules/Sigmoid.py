from modules.ABSLayer import Layer
import numpy as np 

class Sigmoid(Layer):
    def __init__(self):
        pass
    
    def forward(self, input):
        """Apply elementwise ReLU to [batch, input_units] matrix"""
        return 1 / (1 + np.exp(-input))

    def backward(self, input, grad_output):
        return grad_output* self.forward(input) * (1 - self.forward(input))