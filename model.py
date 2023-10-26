import os
import torchvision.transforms as transforms

from torchvision import datasets

import torch

from modules.generator import Generator
from modules.discriminator import Discriminator



def build_model(opt):
    # Loss function
    criterion = torch.nn.BCELoss()

    img_shape = (opt.channels, opt.img_size, opt.img_size)

    # Initialize generator and discriminator
    G = Generator(img_shape, opt.latent_dim)
    D = Discriminator(img_shape)

    G.to(opt.device)
    D.to(opt.device)
    criterion.to(opt.device)

    if not os.path.exists('images'):
        os.mkdir('images')

    # Configure data loader
    data_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )

    # Optimizers
    g_optimizer = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    return G, D, g_optimizer, d_optimizer, criterion, data_loader
