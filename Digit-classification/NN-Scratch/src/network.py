import numpy as np
import random

class Network():
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] 
    
    def forward(self, a):
        activations = [a]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            zs.append(z)
            a = relu(z) if len(zs) < len(self.weights) else softmax(z)
            activations.append(a)
        return activations, zs
    
    def backward(self, activations, zs, y): 
        l = self.num_layers - 1

        grads_W = []
        grads_B = []
        
        # dL/dz_n
        delta = activations[l] - y
        dW = np.dot(delta, activations[l-1].T)
        dB = delta
        grads_W.append(dW)
        grads_B.append(dB)
        for l in reversed(range(1, l)):
            delta = np.dot(self.weights[l].T, delta) * relu_derivative(zs[l-1])
            dW = np.dot(delta, activations[l-1].T)
            dB = delta
            grads_W.append(dW)
            grads_B.append(dB)

        return reversed(grads_W), reversed(grads_B)

    
    def fit(self, train_data, epochs, batch_size, learning_rate):
        n = len(train_data)
        for _ in range(epochs):
            random.shuffle(train_data)
            batches = [train_data[k : k + batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.update_batch(batch, batch_size, learning_rate)
            

    def update_batch(self, batch, batch_size, learning_rate):
        grads_W = [np.zeros(w.shape) for w in self.weights]
        grads_B = [np.zeros(b.shape) for b in self.biases]
        for x, y in batch:
            activations, zs = self.forward(x)
            delta_grads_W, delta_grads_B = self.backward(activations, zs, y)
            grads_W = [w + delta_w for w, delta_w in zip(grads_W, delta_grads_W)]
            grads_B = [b + delta_b for b, delta_b in zip(grads_B, delta_grads_B)]

        for l in range(self.num_layers - 1):
            self.weights[l] -= learning_rate * (grads_W[l] / batch_size)
            self.biases[l] -= learning_rate * (grads_B[l] / batch_size)
    
    def predict(self, x):
        activations, _ = self.forward(x)
        return activations[-1]
        
    def test(self, test_data):
        Y = []
        y_hat = []
        for x, y in test_data:
            Y.append(y)
            y_hat.append(self.predict(x))
        Y = np.array(Y)
        y_hat = np.array(y_hat)
        print(f"Accuracy is: {cross_entropy(Y, y_hat)}")



def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def cross_entropy(y, y_hat):
   return -np.sum(y * np.log(y_hat + 1e-8)) / y.shape[0]