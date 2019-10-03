import h5py
import numpy as np

from random import randint

# load mnist
MNIST_data = h5py.File('data/MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:, 0], ndmin=2))
x_test = np.float32(MNIST_data['x_test'][:])
y_test = np.int32(np.array(MNIST_data['y_test'][:, 0], ndmin=2))
MNIST_data.close()

# one-hot encoding
y = np.hstack((y_train, y_test)).reshape(-1)
y_enc = np.eye(10)[y]
y_train, y_test = y_enc[:60000, :], y_enc[60000:, :]


# Activation functions
def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


# Activation function derivatives
def relu_derivative(p):
    p[p <= 0] = 0
    p[p > 0] = 1
    return p


def sigmoid_derivative(z):
    p = sigmoid(z)
    return p*(1 - p)


# Softmax
def softmax(z):
    Z = np.exp(z)/np.sum(np.exp(z), axis=0)
    return Z


# Class definition
class NeuralNetwork:
    def __init__(self, input_dim, output_dim, hidden_dim, lr, batch_size):
        self.lr = lr
        self.batch_size = batch_size

        # init weights
        self.weights = {"W": np.random.randn(hidden_dim, input_dim)*np.sqrt(1/input_dim),
                        "b1": np.zeros((hidden_dim, 1))*np.sqrt(1/input_dim),
                        "C": np.random.randn(out_dim, hidden_dim)*np.sqrt(1/hidden_dim),
                        "b2": np.zeros((out_dim, 1))*np.sqrt(1/hidden_dim)}

        # set activation function
        self.act_func = relu
        self.dact_func = relu_derivative

        # storage for activations and gradients
        self.activ = {}
        self.grads = {}

    def feedforward(self, input):
         # input to hidden layer
        self.activ["Z"] = np.matmul(self.weights["W"], input.T) + self.weights["b1"]

        # activation
        self.activ["H"] = self.act_func(self.activ["Z"])

        # hidden to output
        self.activ["U"] = np.matmul(self.weights["C"], self.activ["H"]) + self.weights["b2"]

        # softmax
        self.output = softmax(self.activ["U"])


    def get_loss(self, target):
        total = np.sum(np.multiply(target, np.log(self.output.T)))

        return -(1/target.shape[0])*total

    def backprop(self):
        # error at output
        dU = self.output - self.target.T

        # gradients for hidden to output
        dC = (1/self.batch_size)*np.matmul(dU, self.activ["H"].T)
        db2 = (1/self.batch_size)*np.sum(dU, axis=1, keepdims=True)

        # back prop error
        delta = np.matmul(self.weights["C"].T, dU)
        sig_prime = delta*self.dact_func(self.activ["Z"])

        # gradients for input to hidden
        db1 = (1/self.batch_size)*np.sum(sig_prime, axis=1, keepdims=True)
        dW = (1/self.batch_size)*np.matmul(sig_prime, self.input)

        self.grads = {"dW": dW, "db1": db1, "dC": dC, "db2": db2}


    def train(self, x_train, y_train, x_test, y_test, epoch_num):
        # determine number of batches in training set
        batch_num = x_train.shape[0] // self.batch_size
        
        for epoch in range(epoch_num):
            # Learning rate schedule
            # decrease by factor of ten 5 times
            if ((epoch % (epoch_num // 5)) == 0):
                self.lr /= 10

            # shuffle training set to randomize batches
            order = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[order, :]
            y_train_shuffled = y_train[order, :]

            for batch in range(batch_num):
                # gather batch
                start_idx = batch*self.batch_size
                end_idx = start_idx + self.batch_size
                self.input = x_train_shuffled[start_idx:end_idx, :]
                self.target = y_train_shuffled[start_idx:end_idx, :]

                # feed forward and back prop for batch
                self.feedforward(self.input)
                self.backprop()

                # sgd
                self.weights["W"] = self.weights["W"] - self.lr*self.grads["dW"]
                self.weights["C"] = self.weights["C"] - self.lr*self.grads["dC"]
                self.weights["b1"] = self.weights["b1"] - self.lr*self.grads["db1"]
                self.weights["b2"] = self.weights["b2"] - self.lr*self.grads["db2"]

            # find loss for training set
            self.feedforward(x_train)
            train_loss = self.get_loss(y_train)

            # find loss for test set
            self.feedforward(x_test)
            test_loss = self.get_loss(y_test)

            # print loss for each epoch
            print("Epoch {}: training loss = {}, test loss = {}".format(epoch + 1, train_loss, test_loss))

    def run(self, input):
        # non vectorized feedforward 
        # input to hidden layer
        Z = np.dot(self.weights["W"], input) + self.weights["b1"]

        # activation
        H = self.act_func(Z)

        # hidden to output
        U = np.matmul(self.weights["C"], H) + self.weights["b2"]

        # softmax
        return softmax(U)

    def test(self, x_test, y_test):
        total_correct = 0
        for n in range(x_test.shape[0]):
            y = np.argmax(y_test[n, :])
            x = x_test[n, :].reshape(784, 1)
            prediction = np.argmax(self.run(x))
            if (prediction == y):
                total_correct += 1
        print("Test accuracy = {}".format(total_correct/np.float(x_test.shape[0])))


# create net
img_dim = 28 * 28
out_dim = 10
hid_dim = 100
alpha = .5
epoch_num = 25
batch_size = 25
NN = NeuralNetwork(img_dim, out_dim, hid_dim, alpha, batch_size)

# train net
NN.train(x_train, y_train, x_test, y_test, epoch_num)

# test net
NN.test(x_test, y_test)