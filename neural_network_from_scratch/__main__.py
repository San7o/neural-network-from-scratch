import numpy as np
import nnfs
from nnfs.datasets import spiral_data # dataset generator

class Activation_ReLU:
    """
    Activation Rectified Linear Unit
    """
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Layer_Dense:
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
    X, y = spiral_data(100, 3)

    layer1 = Layer_Dense(2, 5)
    activation1 = Activation_ReLU()

    layer1.forward(X)
    activation1.forward(layer1.output)

    print(activation1.output)

if __name__ == "__main__":
    main()
