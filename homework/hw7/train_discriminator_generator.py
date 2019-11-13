import time
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from GAN_model import Discriminator, Generator


# Wasserstein GAN gradient penalty
def calc_gradient_penalty(netD, real_data, fake_data, batch_size):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size,
                         int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    alpha = alpha.to("cuda:0")

    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha*real_data.detach() + ((1 - alpha)*fake_data.detach())

    interpolates = interpolates.to("cuda:0")
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to("cuda:0"),
                              create_graph=True, retain_graph=True,
                              only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


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
    # transform functions for training and test data
    # augment and normalize training set
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
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
    batch_size = 100
    n_classes = 10
    n_z = 100
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True,
                                            transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True,
                                           transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size,
                                             shuffle=False, num_workers=2)

    # copy models to GPU
    aD = Discriminator()
    aG = Generator()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    aD.to(device)
    aG.to(device)
    epoch = 0

    # define the loss function, optimizer, and set learning rate schedule
    criterion = nn.CrossEntropyLoss()
    optimizer_d = optim.Adam(aD.parameters(), lr=.0001, betas=(0, .9))
    optimizer_g = optim.Adam(aG.parameters(), lr=.0001, betas=(0, .9))
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d,
                                            step_size=100, gamma=.1)
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g,
                                            step_size=100, gamma=.1)

    # noise for generator
    np.random.seed(352)
    label = np.asarray(list(range(10))*10)
    noise = np.random.normal(0, 1, (100, n_z))
    label_onehot = np.zeros((100, n_classes))
    label_onehot[np.arange(100), label] = 1
    noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
    noise = noise.astype(np.float32)

    save_noise = torch.from_numpy(noise)
    save_noise = save_noise.to(device)

    # loss and test accuracy storage
    grad_penalties = []
    f_source_losses = []
    r_source_losses = []
    r_class_losses = []
    f_class_losses = []
    train_accuracies = []
    val_accuracies = []

    # control how often generator gets trained
    gen_train = 1

    start_time = time.time()

    # train the models
    num_epochs = 250
    for epoch in range(epoch, epoch + num_epochs):
        # train phase
        aD.train()
        aG.train()

        # for calculating accuracy and losses at each epoch
        loss1 = []
        loss2 = []
        loss3 = []
        loss4 = []
        loss5 = []
        acc1 = []

        time1 = time.time()

        for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader, 0):
            if(Y_train_batch.shape[0] < batch_size):
                continue

            # train generator
            if((batch_idx % gen_train) == 0):
                for p in aD.parameters():
                    p.requires_grad_(False)

                aG.zero_grad()

                label = np.random.randint(0, n_classes, batch_size)
                noise = np.random.normal(0, 1, (batch_size, n_z))
                label_onehot = np.zeros((batch_size, n_classes))
                label_onehot[np.arange(batch_size), label] = 1
                noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
                noise = noise.astype(np.float32)
                noise = torch.from_numpy(noise).to(device)
                fake_label = torch.from_numpy(label).type(torch.long).to(device)

                fake_data = aG(noise)
                gen_source, gen_class = aD(fake_data)

                gen_source = gen_source.mean()
                gen_class = criterion(gen_class, fake_label)

                gen_cost = -gen_source + gen_class
                gen_cost.backward()

                optimizer_g.step()

            # train discriminator
            for p in aD.parameters():
                p.requires_grad_(True)

            aD.zero_grad()

            # train discriminator with generated data
            label = np.random.randint(0, n_classes, batch_size)
            noise = np.random.normal(0, 1, (batch_size, n_z))
            label_onehot = np.zeros((batch_size, n_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = noise.to(device)
            fake_label = torch.from_numpy(label).type(torch.long).to(device)
            with torch.no_grad():
                fake_data = aG(noise)

            disc_fake_source, disc_fake_class = aD(fake_data)

            disc_fake_source = disc_fake_source.mean()
            disc_fake_class = criterion(disc_fake_class, fake_label)

            # train discriminator with real data
            real_data = X_train_batch.to(device)
            real_label = Y_train_batch.to(device)

            disc_real_source, disc_real_class = aD(real_data)

            _, preds = torch.max(disc_real_class.data, 1)
            accuracy = (float(preds.eq(real_label.data).sum())/float(batch_size))*100.0

            disc_real_source = disc_real_source.mean()
            disc_real_class = criterion(disc_real_class, real_label)

            gradient_penalty = calc_gradient_penalty(aD, real_data,
                                                     fake_data, batch_size)

            disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
            disc_cost.backward()

            optimizer_d.step()

            # update average loss
            loss1.append(gradient_penalty.item())
            loss2.append(disc_fake_source.item())
            loss3.append(disc_real_source.item())
            loss4.append(disc_real_class.item())
            loss5.append(disc_fake_class.item())
            acc1.append(accuracy)
            if((batch_idx % 50) == 0):
                print(epoch, batch_idx, "%.2f" % np.mean(loss1),
                                        "%.2f" % np.mean(loss2),
                                        "%.2f" % np.mean(loss3),
                                        "%.2f" % np.mean(loss4),
                                        "%.2f" % np.mean(loss5),
                                        "%.2f" % np.mean(acc1))

        train_accuracies.append(np.mean(acc1))
        grad_penalties.append(np.mean(loss1))
        f_source_losses.append(np.mean(loss2))
        r_source_losses.append(np.mean(loss3))
        r_class_losses.append(np.mean(loss4))
        f_class_losses.append(np.mean(loss5))

        print(epoch, "%.2f" % np.mean(acc1), "%.4f" % np.mean(loss2),
                                             "%.4f" % np.mean(loss3),
                                             "%.4f" % np.mean(loss4),
                                             "%.4f" % np.mean(loss5),
                                             "%.4f" % float(time.time()-time1))

        scheduler_d.step()
        scheduler_g.step()

        # validation phase
        aD.eval()

        # for calculating accuracy at each epoch
        correct = 0.
        total = 0.

        time2 = time.time()

        # test current model
        with torch.no_grad():
            for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):

                # copy the test images to the GPU
                inputs, labels = X_test_batch.to(device), Y_test_batch.to(device)

                # feedforward
                _, outputs = aD(inputs)

                # make predictions
                _, preds = torch.max(outputs.data, 1)

                # get accuracy
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_accuracy = float(correct/total) * 100
        val_accuracies.append(val_accuracy)
        print("Test accuracy: %.2f" % val_accuracy,
              "%.4f" % float(time.time()-time2))

        # save examples of generator outputs
        with torch.no_grad():
            aG.eval()
            samples = aG(save_noise)
            samples = samples.data.to("cpu").numpy()
            samples += 1.0
            samples /= 2.0
            samples = samples.transpose(0, 2, 3, 1)
            aG.train()

        fig = plot(samples)
        plt.savefig('%s.png' % str(epoch).zfill(3), bbox_inches='tight')
        plt.close(fig)

    torch.save(aG, 'generator.model')
    torch.save(aD, 'discriminator.model')

    print("Total training time: %.4f" % float(time.time() - start_time))

    # save results
    with open('grad_penalty{:03d}.pkl'.format(epoch), 'wb') as handle:
        pickle.dump(grad_penalties, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('f_source_loss{:03d}.pkl'.format(epoch), 'wb') as handle:
        pickle.dump(f_source_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('r_source_loss{:03d}.pkl'.format(epoch), 'wb') as handle:
        pickle.dump(r_source_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('f_class_loss{:03d}.pkl'.format(epoch), 'wb') as handle:
        pickle.dump(f_class_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('r_class_loss{:03d}.pkl'.format(epoch), 'wb') as handle:
        pickle.dump(r_class_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('GAN_train_accuracy{:03d}.pkl'.format(epoch), 'wb') as handle:
        pickle.dump(train_accuracies, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('GAN_val_accuracy{:03d}.pkl'.format(epoch), 'wb') as handle:
        pickle.dump(val_accuracies, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save checkpoint
    checkpoint_filename = 'homework/hw7/checkpoints/GAN_cifar10_{:03d}.tar'.format(epoch)
    torch.save({
            'epoch': epoch,
            'aD_state_dict': aD.state_dict(),
            'aG_state_dict': aG.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'scheduler_d_state_dict': scheduler_d.state_dict(),
            'scheduler_g_state_dict': scheduler_g.state_dict(),
            }, checkpoint_filename)
