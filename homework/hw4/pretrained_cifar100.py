import torch
import torchvision
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms


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

if __name__=="__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    train_dataset = torchvision.datasets.CIFAR100(root='../data', train=True,
                                            download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256,
                                            shuffle=True, num_workers=8)

    val_dataset = torchvision.datasets.CIFAR100(root='../data', train=False,
                                        download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256,
                                            shuffle=False, num_workers=8)

    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc = nn.Linear(512, 100)
    model = nn.Sequential(nn.Upsample(scale_factor=7, mode='bilinear'),
                        resnet18)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=.1, weight_decay=5e-4, momentum=.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    train_losses, train_accuracies, valid_losses, test_accuracies = train(model, criterion, optimizer, scheduler, num_epochs=20)
