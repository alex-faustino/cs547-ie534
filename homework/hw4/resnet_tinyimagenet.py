import os
import math
import multiprocessing

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


def create_val_folder(val_dir):
  """
  This method is responsible for separating validation
  images into separate sub folders
  """
  # path where validation data is present now
  path = os.path.join(val_dir, 'images')
  # file where image2class mapping is present
  filename = os.path.join(val_dir, 'val_annotations.txt')
  fp = open(filename, "r") # open file in read mode
  data = fp.readlines() # read line by line

  """
  Create a dictionary with image names as key and
  corresponding classes as values
  """
  val_img_dict = {}
  for line in data:
    words = line.split("\t")
    val_img_dict[words[0]] = words[1]
  fp.close()
  # Create folder if not present, and move image into proper folder
  for img, folder in val_img_dict.items():
    newpath = (os.path.join(path, folder))
    if not os.path.exists(newpath): # check if folder exists
      os.makedirs(newpath)
    # Check if image exists in default directory
    if os.path.exists(os.path.join(path, img)):
      os.replace(os.path.join(path, img), os.path.join(newpath, img))
  return

# create_val_folder("D:/Users/Penti/cs547/data/tiny-imagenet-200/val")

if __name__ == '__main__':
    # set device to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # transform functions for training and test data
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # augment and normalize training set
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    # normalize test set
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_dir = "D:/Users/Penti/cs547/data/tiny-imagenet-200/train"
    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)

    val_dir = "D:/Users/Penti/cs547/data/tiny-imagenet-200/val/images"
    val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8)

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

    # define the model
    # Structure based off https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html
    # Convolution layer template
    def conv3x3(in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        bias=False)
    

    class BasicBlock(nn.Module):
        """Basic Block for ResNet"""

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
        """ResNet builder"""

        def __init__(self, block, reps, num_classes=200):
            super(ResNet, self).__init__()
            self.in_channels = 32

            # pre basic blocks convolution with dropout
            self.conv1 = conv3x3(in_channels=3, out_channels=32)
            self.bn = nn.BatchNorm2d(32)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout2d(p=0.2)
        
            # basic blocks
            self.layer1 = self._make_layer(block, 32, reps[0])
            self.layer2 = self._make_layer(block, 64, reps[1], stride=2)
            self.layer3 = self._make_layer(block, 128, reps[2], stride=2)
            self.layer4 = self._make_layer(block, 256, reps[3], stride=2)

            # pooling, fully connected, and softmax
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc = nn.Linear(256*4, num_classes)
            self.softmax = nn.Softmax(dim=1)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()

        def _make_layer(self, block, out_channels, reps, stride=1):
            downsample = None
            if stride != 1 or self.in_channels != out_channels:
                downsample = nn.Sequential(
                    conv3x3(self.in_channels, out_channels, stride),
                    nn.BatchNorm2d(out_channels)
                )

            layers = []
            layers.append(block(self.in_channels, out_channels, stride, downsample))
            self.in_channels = out_channels
            for _ in range(1, reps):
                layers.append(block(self.in_channels, out_channels))

            return nn.Sequential(*layers)

        def forward(self, x):
            out = nn.functional.interpolate(x, size=(32, 32))
            out = self.conv1(out)
            out = self.bn(out)
            out = self.relu(out)
            out = self.dropout(out)

            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

            out = self.maxpool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)

            return out

    # train the model
    def train(model, criterion, optimizer, scheduler, num_epochs, first_epoch=1):

        train_losses = []
        train_accuracies = []
        valid_losses = []
        test_accuracies = []

        for epoch in range(first_epoch, first_epoch + num_epochs):
            print('Epoch', epoch)

            # train phase
            model.train()

            train_loss = MovingAverage()

            # for calculating accuracy at each epoch
            train_correct = 0
            train_total = 0

            for i, (images, labels) in enumerate(train_loader):
                # Move the training data to the GPU
                images, labels = images.to(device), labels.to(device)

                # clear previous gradient computation
                optimizer.zero_grad()

                # forward propagation
                outputs = model(images)

                # calculate the loss
                loss = criterion(outputs, labels)

                # backpropagate to compute gradients
                loss.backward()

                # update model weights
                optimizer.step()

                # update average loss
                train_loss.update(loss)

                # make predictions
                _, train_preds = torch.max(outputs.data, 1)

                # Calculate training accuracy
                train_total += labels.size(0)
                train_correct += (train_preds == labels.data).sum().item()

            print('Training loss:', train_loss)
            train_losses.append(train_loss.value)

            train_accuracy = float(train_correct/train_total) * 100
            print('Train accuracy: {:.4f}%'.format(train_accuracy))
            train_accuracies.append(train_accuracy)

            # validation phase
            model.eval()

            valid_loss = RunningAverage()

            # for calculating accuracy at each epoch
            val_correct = 0
            val_total = 0

            # test current model
            with torch.no_grad():
                for images, labels in val_loader:

                    # Move the test images to the GPU
                    images, labels = images.to(device), labels.to(device)

                    # forward propagation
                    outputs = model(images)

                    # calculate the loss
                    loss = criterion(outputs, labels)

                    # update running loss value
                    valid_loss.update(loss)

                    # make predictions
                    _, val_preds = torch.max(outputs.data, 1)

                    # raise ValueError('Time to debug.')
                    # Calculate validation accuracy
                    val_total += labels.size(0)
                    val_correct += (val_preds == labels.data).sum().item()

            print('Validation loss:', valid_loss)
            valid_losses.append(valid_loss.value)

            test_accuracy = float(val_correct/val_total) * 100
            print('Test accuracy: {:.4f}%'.format(test_accuracy))
            test_accuracies.append(test_accuracy)

            # Update learning rate
            scheduler.step(loss)
    
        return train_losses, train_accuracies, valid_losses, test_accuracies

    # create model object and copy it to gpu
    model = ResNet(BasicBlock,(2,4,4,2))
    model.to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=.01, weight_decay=1e-3, momentum=.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=.1)

    train_losses, train_accuracies, valid_losses, test_accuracies = train(model, criterion, optimizer, scheduler, num_epochs=60)
