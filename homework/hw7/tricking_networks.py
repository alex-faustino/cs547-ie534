import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

matplotlib.use('Agg')


def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig


if __name__ == '__main__':
    transform_test = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    batch_size = 128
    testset = torchvision.datasets.CIFAR10(root='../data',
                                           train=False,
                                           download=False,
                                           transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8)
    testloader = enumerate(testloader)

    model = torch.load('homework/hw7/cifar10.model')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    batch_idx, (X_batch, Y_batch) = next(testloader)
    X_batch = X_batch.to(device)
    X_batch.requires_grad = True
    Y_batch_alternate = (Y_batch + 1) % 10
    Y_batch_alternate = Y_batch_alternate.to(device)
    Y_batch = Y_batch.to(device)

    # save real images
    samples = X_batch.data.to("cpu").numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0, 2, 3, 1)

    fig = plot(samples[0:100])
    plt.savefig('homework/hw7/visualization/real_images.png',
                bbox_inches='tight')
    plt.close(fig)

    _, output = model(X_batch)
    preds = output.data.max(1)[1]  # first column has actual prob.
    accuracy = (float(preds.eq(Y_batch.data).sum())/float(batch_size))*100.0
    print(accuracy)

    # slightly jitter all input images
    criterion = nn.CrossEntropyLoss(reduce=False)
    loss = criterion(output, Y_batch_alternate)

    gradients = torch.autograd.grad(outputs=loss,
                                    inputs=X_batch,
                                    grad_outputs=torch.ones(loss.size()).to(device),
                                    create_graph=True,
                                    retain_graph=False,
                                    only_inputs=True)[0]

    # save gradient jitter
    gradient_image = gradients.data.to("cpu").numpy()
    gradient_image = (gradient_image - np.min(gradient_image))/(np.max(gradient_image) - np.min(gradient_image))
    gradient_image = gradient_image.transpose(0, 2, 3, 1)
    fig = plot(gradient_image[0:100])
    plt.savefig('homework/hw7/visualization/gradient_image.png',
                bbox_inches='tight')
    plt.close(fig)

    # jitter input image
    gradients[gradients > 0.0] = 1.0
    gradients[gradients < 0.0] = -1.0

    gain = 8.0
    X_batch_modified = X_batch - gain*0.007843137*gradients
    X_batch_modified[X_batch_modified > 1.0] = 1.0
    X_batch_modified[X_batch_modified < -1.0] = -1.0

    # evaluate new fake images
    _, output = model(X_batch_modified)
    preds = output.data.max(1)[1]  # first column has actual prob.
    accuracy = (float(preds.eq(Y_batch.data).sum())/float(batch_size))*100.0
    print(accuracy)

    # save fake images
    samples = X_batch_modified.data.to("cpu").numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0, 2, 3, 1)

    fig = plot(samples[0:100])
    plt.savefig('homework/hw7/visualization/jittered_images.png',
                bbox_inches='tight')
    plt.close(fig)
