import numpy as np

# Take an input, dot product it with the weights, add the biases
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # We want to normalize the random weights
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

def main():
    # Input Batch
    X = [ [1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8] ]

    np.random.seed(0)

    layer1 = Layer_Dense(4, 5)
    layer2 = Layer_Dense(5, 2)

    layer1.forward(X)
    print(layer1.output)
    layer2.forward(layer1.output)
    print(layer2.output)

if __name__ == "__main__":
    main()
