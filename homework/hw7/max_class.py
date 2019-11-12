import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

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

    model = torch.load('homework/hw7/discriminator.model')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    batch_idx, (X_batch, Y_batch) = next(testloader)
    X_batch = X_batch.to(device)
    X_batch.requires_grad = True

    X = X_batch.mean(dim=0)
    X = X.repeat(10, 1, 1, 1)

    Y = torch.arange(10).type(torch.int64)
    Y = Y.to(device)

    lr = 0.1
    weight_decay = 0.001
    for i in range(200):
        _, output = model(X)

        loss = -output.diag()
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                        grad_outputs=torch.ones(loss.size()).to(device),
                                        create_graph=True, retain_graph=False,
                                        only_inputs=True)[0]

        preds = output.data.max(1)[1]  # first column has actual prob.
        accuracy = (float(preds.eq(Y.data).sum())/float(10.0))*100.0
        print(i, accuracy, -loss)

        X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
        X[X > 1.0] = 1.0
        X[X < -1.0] = -1.0

    # save new images
    samples = X.data.to("cpu").numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0, 2, 3, 1)

    fig = plot(samples)
    plt.savefig('homework/hw7/visualization/max_class_gen.png', bbox_inches='tight')
    plt.close(fig)
