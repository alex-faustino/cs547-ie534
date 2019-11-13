import torch.nn as nn
import torch.nn.functional as F


# convolution layer template
def convTemp(in_channels, out_channels, k_size, padding=1, stride=1):
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=k_size,
                     stride=stride,
                     padding=padding,
                     bias=False)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # convolution layers
        self.conv1 = convTemp(3, 196, 3, 1, 1)
        self.conv2 = convTemp(196, 196, 3, 1, 2)
        self.conv3 = convTemp(196, 196, 3, 1, 1)
        self.conv4 = convTemp(196, 196, 3, 1, 2)
        self.conv5 = convTemp(196, 196, 3, 1, 1)
        self.conv6 = convTemp(196, 196, 3, 1, 1)
        self.conv7 = convTemp(196, 196, 3, 1, 1)
        self.conv8 = convTemp(196, 196, 3, 1, 2)

        # layer normalizations
        self.ln1 = nn.LayerNorm([32, 32])
        self.ln2 = nn.LayerNorm([16, 16])
        self.ln3 = nn.LayerNorm([16, 16])
        self.ln4 = nn.LayerNorm([8, 8])
        self.ln5 = nn.LayerNorm([8, 8])
        self.ln6 = nn.LayerNorm([8, 8])
        self.ln7 = nn.LayerNorm([8, 8])
        self.ln8 = nn.LayerNorm([4, 4])

        # leaky relu
        self.leaky_relu = nn.LeakyReLU()

        # pool layer
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4)

        # critic
        self.fc1 = nn.Linear(196, 1)

        # auxiliary classifier
        self. fc10 = nn.Linear(196, 10)

    def forward(self, x, extract_features=8):
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.ln2(x)
        x = self.leaky_relu(x)

        x = self.conv3(x)
        x = self.ln3(x)
        x = self.leaky_relu(x)

        x = self.conv4(x)
        x = self.ln4(x)
        x = self.leaky_relu(x)
        if extract_features == 4:
            x = F.max_pool2d(x, 8, 8)
            x = x.view(-1, 196)

            return x

        x = self.conv5(x)
        x = self.ln5(x)
        x = self.leaky_relu(x)

        x = self.conv6(x)
        x = self.ln6(x)
        x = self.leaky_relu(x)

        x = self.conv7(x)
        x = self.ln7(x)
        x = self.leaky_relu(x)

        x = self.conv8(x)
        x = self.ln8(x)
        x = self.leaky_relu(x)
        if extract_features == 8:
            x = F.max_pool2d(x, 4, 4)
            x = x.view(-1, 196)

            return x

        x = self.pool(x)

        x = x.view(x.size(0), -1)
        critic = self.fc1(x)
        output = self.fc10(x)

        return critic, output


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # input layer
        self.fc1 = nn.Linear(100, 196*4*4)

        # convolution layers
        self.conv1 = nn.ConvTranspose2d(in_channels=196,
                                        out_channels=196,
                                        kernel_size=4,
                                        padding=1,
                                        stride=2)
        self.conv2 = convTemp(196, 196, 3, 1, 1)
        self.conv3 = convTemp(196, 196, 3, 1, 1)
        self.conv4 = convTemp(196, 196, 3, 1, 1)
        self.conv5 = nn.ConvTranspose2d(in_channels=196,
                                        out_channels=196,
                                        kernel_size=4,
                                        padding=1,
                                        stride=2)
        self.conv6 = convTemp(196, 196, 3, 1, 1)
        self.conv7 = nn.ConvTranspose2d(in_channels=196,
                                        out_channels=196,
                                        kernel_size=4,
                                        padding=1,
                                        stride=2)
        self.conv8 = convTemp(196, 3, 3, 1, 1)

        # batch normalizations
        self.bn0 = nn.BatchNorm1d(196*4*4)
        self.bn1 = nn.BatchNorm2d(196)
        self.bn2 = nn.BatchNorm2d(196)
        self.bn3 = nn.BatchNorm2d(196)
        self.bn4 = nn.BatchNorm2d(196)
        self.bn5 = nn.BatchNorm2d(196)
        self.bn6 = nn.BatchNorm2d(196)
        self.bn7 = nn.BatchNorm2d(196)

        # activations
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn0(x)
        x = x.view(-1, 196, 4, 4)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self. relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self. relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self. relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self. relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self. relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self. relu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self. relu(x)
        x = self.conv8(x)
        out = self.tanh(x)

        return out
