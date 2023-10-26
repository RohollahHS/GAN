import numpy as np
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,  opt):
        super(Generator, self).__init__()
        image_size = opt.image_size
        latent_size = opt.latent_size
        hidden_size = opt.hidden_size

        self.model = nn.Sequential(
                    nn.Linear(latent_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, image_size),
                    nn.Tanh())

    def forward(self, z):
        img = self.model(z)
        return img
