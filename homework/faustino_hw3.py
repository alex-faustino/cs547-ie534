import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn


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


# save and load checkpoints in training
def save_checkpoint(optimizer, model, epoch, filename):
    checkpoint_dict = {
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint_dict, filename)


def load_checkpoint(optimizer, model, filename):
    checkpoint_dict = torch.load(filename)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch


# transform functions for training and test data
# augment and normalize training set
# normalization values taken from https://github.com/kuangliu/pytorch-cifar/issues/19
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

# normalize test set
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

# load dataset
# taken from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# define net using torch.nn sequential container
model = nn.Sequential(
        # blocks are separated by pooling, dropout, or both
        # conv block 1
        nn.Conv2d(in_channels=3,
                  out_channels=32,
                  kernel_size=3,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(32),
        nn.Conv2d(in_channels=32,
                  out_channels=32,
                  kernel_size=3,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(p=.2),

        # conv block 2
        nn.Conv2d(in_channels=32,
                  out_channels=64,
                  kernel_size=3,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.Conv2d(in_channels=64,
                  out_channels=64,
                  kernel_size=3,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(p=.3),

        # conv block 3
        nn.Conv2d(in_channels=64,
                  out_channels=128,
                  kernel_size=3,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.Conv2d(in_channels=128,
                  out_channels=128,
                  kernel_size=3),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(p=.4),

        # flatten
        nn.Flatten(),

        # fully connected layers
        nn.Linear(in_features=1152, out_features=576),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=576, out_features=288),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=288, out_features=10),
        nn.Softmax(dim=1)
)


# copy model to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# define the loss function, optimizer, and set learning rate schedule
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

def train(model, criterion, optimizer, num_epochs, first_epoch=1):
    # loss and test accuracy storage
    train_losses = []
    valid_losses = []
    test_accuracies = []

    for epoch in range(first_epoch, first_epoch + num_epochs):
        print('Epoch', epoch)

        # train phase
        model.train()

        # moving average of the training loss during the epoch
        train_loss = MovingAverage()

        for i, data in enumerate(trainloader, 0):
            # copy the training data to the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero out grad 
            optimizer.zero_grad()

            # feedforward
            outputs = model(inputs)

            # get loss
            loss = criterion(outputs, labels)

            # backprop
            loss.backward()

            # update model
            optimizer.step()

            # update average loss
            train_loss.update(loss)

        print('Training loss: ', train_loss)
        train_losses.append(train_loss.value)


        # validation phase
        model.eval()

        # create running average for losses during validation of the epoch
        valid_loss = RunningAverage()

        # for calculating accuracy at each epoch
        correct = 0.
        total = 0.

        # test current model
        with torch.no_grad():
            for data in testloader:

                # copy the test images to the GPU
                inputs, labels = data[0].to(device), data[1].to(device)

                # feedforward
                outputs = model(inputs)

                # get loss
                loss = criterion(outputs, labels)

                # update average loss
                valid_loss.update(loss)

                # make predictions
                _, predictions = torch.max(outputs.data, 1)

                # get accuracy
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        print('Validation loss: ', valid_loss)
        valid_losses.append(valid_loss.value)

        test_accuracy = float(correct/total) * 100
        print('Test accuracy: {:.4f}%'.format(test_accuracy))
        test_accuracies.append(test_accuracy)

        # update learning rate
        scheduler.step()

        # save a checkpoint for each epoch
        checkpoint_filename = 'checkpoints/cifar10-{:03d}.pkl'.format(epoch)
        save_checkpoint(optimizer, model, epoch, checkpoint_filename)
    
    return train_losses, valid_losses, test_accuracies


# train CNN for 120 epochs
train_losses, valid_losses, test_accuracies = train(model, criterion, optimizer, num_epochs=120)


# plot results
import matplotlib.pyplot as plt

epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(10,6))
plt.plot(epochs, train_losses, '-o', label='Training loss')
plt.plot(epochs, valid_losses, '-o', label='Validation loss')
plt.legend()
plt.title('Learning curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(epochs, test_accuracies, '-o', label='Test accuracy')
plt.legend()
plt.title('Test Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()