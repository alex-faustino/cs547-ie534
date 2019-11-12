import time
import pickle

# import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from GAN_model import Discriminator


# helper classes and functions
# moving average classes taken from: https://colab.research.google.com/drive/1gJAAN3UI9005ecVmxPun5ZLCGu4YBtLo#scrollTo=9QAuxIQvoSDV
class AverageBase(object):
    
    def __init__(self, value=0):
        self.value = float(value) if value is not None else None
       
    def __str__(self):
        return str(round(self.value, 4))
    
    def __repr__(self):
        return self.value
    
    def __format__(self, fmt):
        return self.value.__format__(fmt)
    
    def __float__(self):
        return self.value


class RunningAverage(AverageBase):
    """
    Keeps track of a cumulative moving average (CMA).
    """
    
    def __init__(self, value=0, count=0):
        super(RunningAverage, self).__init__(value)
        self.count = count
        
    def update(self, value):
        self.value = (self.value * self.count + float(value))
        self.count += 1
        self.value /= self.count
        return self.value


class MovingAverage(AverageBase):
    """
    An exponentially decaying moving average (EMA).
    """
    
    def __init__(self, alpha=0.99):
        super(MovingAverage, self).__init__(None)
        self.alpha = alpha
        
    def update(self, value):
        if self.value is None:
            self.value = float(value)
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * float(value)
        return self.value


if __name__ == '__main__':
    # transform functions for training and test data
    # augment and normalize training set
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
        transforms.ColorJitter(
                brightness=abs(float(0.1*torch.randn(1))),
                contrast=abs(float(0.1*torch.randn(1))),
                saturation=abs(float(0.1*torch.randn(1))),
                hue=abs(float(0.1*torch.randn(1)))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # load dataset
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)

    # copy model to GPU
    model = Discriminator()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # define the loss function, optimizer, and set learning rate schedule
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # loss and test accuracy storage
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    for epoch in range(40):
        scheduler.step()

        # train phase
        model.train()

        # moving average of the training loss during the epoch
        train_loss = MovingAverage()

        # for calculating accuracy at each epoch
        correct = 0.
        total = 0.

        time1 = time.time()

        for i, data in enumerate(trainloader, 0):
            # copy the training data to the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero out grad 
            optimizer.zero_grad()

            # feedforward
            _, outputs = model(inputs)

            # get loss
            loss = criterion(outputs, labels)

            # backprop
            loss.backward()

            # update model
            optimizer.step()

            # update average loss
            train_loss.update(loss)

            # make predictions
            _, predictions = torch.max(outputs.data, 1)

            # get accuracy
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

        train_accuracy = float(correct/total) * 100
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss.value)
        print(epoch, "%.2f" % train_accuracy, "%.4f" % train_loss, "%.4f" % float(time.time()-time1))


        # validation phase
        model.eval()

        # create running average for losses during validation of the epoch
        valid_loss = RunningAverage()

        # for calculating accuracy at each epoch
        correct = 0.
        total = 0.

        time2 = time.time()

        # test current model
        with torch.no_grad():
            for data in testloader:

                # copy the test images to the GPU
                inputs, labels = data[0].to(device), data[1].to(device)

                # feedforward
                _, outputs = model(inputs)

                # get loss
                loss = criterion(outputs, labels)

                # update average loss
                valid_loss.update(loss)

                # make predictions
                _, predictions = torch.max(outputs.data, 1)

                # get accuracy
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        valid_accuracy = float(correct/total) * 100
        valid_accuracies.append(valid_accuracy)
        valid_losses.append(valid_loss.value)
        print("%.2f" % valid_accuracy, "%.4f" % valid_loss, "%.4f" % float(time.time()-time2))

    torch.save(model, 'cifar10.model')

    # save results
    with open('cifar10_train_loss.pkl', 'wb') as handle:
        pickle.dump(train_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('cifar10_val_loss.pkl', 'wb') as handle:
        pickle.dump(valid_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('cifar10_train_accuracy.pkl', 'wb') as handle:
        pickle.dump(train_accuracies, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('cifar10_val_accuracy.pkl', 'wb') as handle:
        pickle.dump(valid_accuracies, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # plot results
    # epochs = range(1, len(train_losses) + 1)
    # plt.figure(figsize=(10,6))
    # plt.plot(epochs, train_losses, '-o', label='Training loss')
    # plt.plot(epochs, valid_losses, '-o', label='Validation loss')
    # plt.legend()
    # plt.title('Loss Curves')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()

    # plt.figure(figsize=(10,6))
    # plt.plot(epochs, train_accuracies, '-o', label='Training accuracy')
    # plt.plot(epochs, valid_accuracies, '-o', label='Validation accuracy')
    # plt.legend()
    # plt.title('Accuracy Curves')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.show()
