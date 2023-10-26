import os
import torchvision.transforms as transforms

from torchvision import datasets

import torch

from modules.generator import Generator
from modules.discriminator import Discriminator



def build_model(opt):
    # Loss function
    criterion = torch.nn.BCELoss()

    # Initialize generator and discriminator
    G = Generator(opt)
    D = Discriminator(opt)

    G.to(opt.device)
    D.to(opt.device)
    criterion.to(opt.device)

    if not os.path.exists('images'):
        os.mkdir('images')

    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5),
                                        std=(0.5))])

    # MNIST dataset
    mnist = datasets.MNIST(root='data/mnist',
                            train=True,
                            transform=transform,
                            download=True)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                            batch_size=opt.batch_size, 
                                            shuffle=True)

    # Optimizers
    g_optimizer = torch.optim.Adam(G.parameters(), lr=opt.lr)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=opt.lr)

    return G, D, g_optimizer, d_optimizer, criterion, data_loader
