import numpy as np
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,  img_shape, latent_dim):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, np.prod(img_shape)),
                    nn.Tanh())

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
