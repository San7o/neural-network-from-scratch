import numpy as np
import nnfs
from nnfs.datasets import spiral_data # dataset generator

class Activation_ReLU:
    """
    Activation Rectified Linear Unit

    If the input is less than 0, we set it to 0. Otherwise we keep the value
    We need this to introduce non-linearity in the network
    """
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    """
    Activation Softmax

    The softmax function is used to normalize the output of a layer as probabilities
    """
    def forward(self, inputs):
        # We sybtract the max value to prevent overflow. The result will
        # be from 0 to 1
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize the values
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Layer_Dense:
    """
    A layer of the network
    """
    def __init__(self, n_inputs, n_neurons):
        # We want to normalize the random weights
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    """ 
    Take an input, dot product it with the weights, add the biases
    """
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

def main():

    # Overrides some numpy defaults to make them more consistant
    # on different machines
    nnfs.init()
    X, y = spiral_data(samples=100, classes=3)

    dense1 = Layer_Dense(2, 3)
    activation1 = Activation_ReLU()

    # Output layer
    dense2 = Layer_Dense(3, 3)
    activation2 = Activation_Softmax()

    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    print(activation2.output[:5])

if __name__ == "__main__":
    main()
