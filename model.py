import os
import torchvision.transforms as transforms

from torchvision import datasets

import torch

from modules.generator import Generator
from modules.discriminator import Discriminator



def build_model(opt):
    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    img_shape = (opt.channels, opt.img_size, opt.img_size)

    # Initialize generator and discriminator
    generator = Generator(img_shape, opt.latent_dim)
    discriminator = Discriminator(img_shape)

    generator.to(opt.device)
    discriminator.to(opt.device)
    adversarial_loss.to(opt.device)

    if not os.path.exists('images'):
        os.mkdir('images')

    # Configure data loader
    dataloader = torch.utils.data.DataLoader(
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
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    return generator, discriminator, optimizer_G, optimizer_D, adversarial_loss, dataloader
