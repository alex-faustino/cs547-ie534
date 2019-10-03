import math
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import multiprocessing
multiprocessing.set_start_method('spawn', True)

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
# normalization values taken from https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])

# normalize test set
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])

# load dataset
# taken from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
trainset = torchvision.datasets.CIFAR100(root='../data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR100(root='../data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=8)


# define model structure
# based off https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html
# convolution layer template
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)
    

class BasicBlock(nn.Module):
    """
    Basic Block for ResNet
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet builder
    """
    def __init__(self, block, reps, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 32

        # pre basic blocks convolution with dropout
        self.conv1 = conv3x3(in_channels=3, out_channels=32)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=.2)
        
        # basic blocks
        self.layer1 = self._make_layer(block, 32, reps[0])
        self.layer2 = self._make_layer(block, 64, reps[1], stride=2)
        self.layer3 = self._make_layer(block, 128, reps[2], stride=2)
        self.layer4 = self._make_layer(block, 256, reps[3], stride=2)

        # pooling, fully connected, and softmax
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)
        self.fc = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, block, out_channels, reps, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, reps):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        out = self.softmax(out)

        return out


# create model object
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, criterion, optimizer, scheduler, num_epochs, first_epoch=1):
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

def run():
    model = ResNet(BasicBlock,(2,4,4,2))
    model.to(device)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # train CNN for 120 epochs
    train_losses, valid_losses, test_accuracies = train(model, criterion, optimizer, scheduler, num_epochs=120)

    # plot results
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

if __name__ == '__main__':
    run()