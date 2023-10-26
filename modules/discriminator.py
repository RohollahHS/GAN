import numpy as np
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.image_size = opt.image_size
        hidden_size = self.hidden_size

        self.model = nn.Sequential(
                    nn.Linear(self.image_size, hidden_size),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_size, hidden_size),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid())

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
