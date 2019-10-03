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

# reshape images
shape = (-1, 28, 28)
x_train = x_train.reshape(shape)
x_test  = x_test.reshape(shape)

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


# Zero pad image
def zero_pad(x, pad):
    return np.pad(x, pad, mode='constant')


# Class definition
class ConvNeuralNetwork:
    def __init__(self, x_height, x_width,                 
                filter_num, filter_height, filter_width,
                output_dim,
                lr,
                stride=1, pad=1):

        # instance variables
        self.x_height, self.x_width = x_height, x_width
        self.filter_num, self.filter_height, self.filter_width = filter_num, filter_height, filter_width
        self.output_dim = output_dim
        self.lr = lr
        self.pad = pad

        # find dimension of output from convolution layer
        self.x_height = x_height
        self.z_heigth = int((x_height - filter_height + 2*pad) / stride + 1)
        self.z_width = int((x_width - filter_width + 2*pad) / stride + 1)
        self.hidden_dim = self.filter_num*self.z_heigth*self.z_width

        # init weights
        self.weights = {"K": np.random.randn(self.filter_num, self.filter_height, self.filter_width)*np.sqrt(1/self.filter_num),
                        "W": np.random.randn(self.output_dim, self.hidden_dim)*np.sqrt(1/self.hidden_dim),
                        "b": np.zeros((self.output_dim, 1))*np.sqrt(1/self.hidden_dim)}

        # set activation function
        self.act_func = relu
        self.dact_func = relu_derivative

        # storage for activations and gradients
        self.activ = {}
        self.grads = {}

    def feedforward(self, input):
        # convolution
        # assumes 1 channel and image is zero padded
        self.x_cols = self.arrange_Conv(input)
        K_row = self.weights["K"].reshape(self.filter_num, -1)
        self.activ["Z"] = K_row.dot(self.x_cols).reshape(self.filter_num, self.z_heigth, self.z_width)

        # activation
        self.activ["H"] = self.act_func(self.activ["Z"].reshape(self.hidden_dim, 1))

        # hidden layer
        self.activ["U"] = np.matmul(self.weights["W"], self.activ["H"]) + self.weights["b"]

        # softmax
        self.output = softmax(self.activ["U"])

        return self.output

    def arrange_Conv(self, x_padded):
        # arrange image so that dot products are vectorized
        i0 = np.repeat(np.arange(self.filter_height), self.filter_width)
        i0 = np.tile(i0, 1)
        i1 = np.repeat(np.arange(self.z_heigth), self.z_width)

        j0 = np.tile(np.arange(self.filter_height), self.filter_width)
        j1 = np.tile(np.arange(self.z_heigth), self.z_width)

        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        return x_padded[i, j]

    def get_loss(self, target):
        total = np.sum(np.multiply(target, np.log(self.output.T)))

        return -(1/target.shape[0])*total

    def backprop(self):
        # error at output
        dU = self.output - self.target.reshape(-1, 1)

        # gradients for hidden to output
        dW = np.matmul(dU, self.activ["H"].T)
        db = dU

        # back prop error
        delta = np.dot(self.weights["W"].T, dU).reshape(self.activ["Z"].shape)
        sig_prime = self.dact_func(self.activ["Z"])

        # gradients for input to hidden
        dK = np.matmul(self.x_cols, (sig_prime*delta).reshape(self.filter_num, -1).T)
        dK = dK.T.reshape(self.filter_num, self.filter_height, self.filter_width)

        self.grads = {"dK": dK, "dW": dW, "db": db}


    def train(self, x_train, y_train, x_test, y_test, epoch_num):
        
        for epoch in range(epoch_num):
            # Learning rate schedule
            # decrease by factor of ten 5 times
            if ((epoch % (epoch_num // 5)) == 0):
                self.lr /= 10

            # epoch loss storage to find mean
            loss = []

            # shuffle training set
            order = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[order, :, :]
            y_train_shuffled = y_train[order, :]

            for i in range(x_train.shape[0]):
                # feed forward and back prop for each image
                self.input = zero_pad(x_train_shuffled[i, :, :], self.pad)
                self.target = y_train_shuffled[i, :]
                self.feedforward(self.input)
                self.backprop()

                # sgd
                self.weights["K"] = self.weights["K"] - self.lr*self.grads["dK"]
                self.weights["W"] = self.weights["W"] - self.lr*self.grads["dW"]
                self.weights["b"] = self.weights["b"] - self.lr*self.grads["db"]

                # find loss for training set
                output = self.feedforward(self.input)
                loss.append(self.get_loss(self.target))

            train_loss = np.mean(loss)

            # print mean loss for each epoch
            print("Epoch {}: training loss = {}".format(epoch + 1, train_loss))

            # test accuracy for each epoch  
            self.test(x_test, y_test)

    def test(self, x_test, y_test):
        total_correct = 0
        for n in range(x_test.shape[0]):
            y = np.argmax(y_test[n, :])
            x = zero_pad(x_test[n, :, :], self.pad)
            prediction = np.argmax(self.feedforward(x))
            if (prediction == y):
                total_correct += 1
        print("Test accuracy = {}".format(total_correct/np.float(x_test.shape[0])))


# Net parameters and create net
x_height, x_width = 28, 28
filter_num, filter_height, filter_width = 16, 7, 7
output_dim = 10
init_lr = 0.5
batch_size = 1
CNN = ConvNeuralNetwork(x_height, x_width,
                        filter_num, filter_height, filter_width,
                        output_dim, init_lr, batch_size)

# train and test net
epoch_num = 5
CNN.train(x_train, y_train, x_test, y_test, epoch_num)