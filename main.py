import argparse
import numpy as np
from model import build_model

from torchvision.utils import save_image

from torch.autograd import Variable
import torch


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")

    opt = parser.parse_args()

    print('Args:')
    for k, v in vars(opt).items():
        print(f'{k}: {v}')
    print()

    return opt


def train(opt):
    device = opt.device
    num_epochs = opt.n_epochs

    d_losses = np.zeros(num_epochs)
    g_losses = np.zeros(num_epochs)
    real_scores = np.zeros(num_epochs)
    fake_scores = np.zeros(num_epochs)

    def reset_grad():
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()

    total_step = len(dataloader)
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(dataloader):
            images = Variable(images)

            # Adversarial ground truths
            real_labels = Variable(torch.ones(images.size(0), 1).to(device))
            fake_labels = Variable(torch.zeros(images.size(0), 1).to(device))

            # ---------------------
            # Train Discriminator on Real Data
            # ---------------------
            outputs  = discriminator(images)
            real_loss = adversarial_loss(outputs, real_labels)
            real_score = outputs

            # ---------------------
            # Train Discriminator on Fake Data
            # ---------------------
            z = Variable(torch.randn(images.shape[0], opt.latent_dim).to(device))

            fake_images = generator(z)

            outputs = discriminator(fake_images)
            fake_loss = adversarial_loss(outputs, fake_labels)
            fake_score = outputs

            d_loss = (real_loss + fake_loss)

            reset_grad()
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            z = Variable(torch.randn(images.shape[0], opt.latent_dim).to(device))
            fake_images = generator(z)
            outputs = discriminator(fake_images)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(outputs, real_labels)

            reset_grad()
            g_loss.backward()
            optimizer_G.step()


            # =================================================================== #
            #                          Update Statistics                          #
            # =================================================================== #
            d_losses[epoch] = d_losses[epoch]*(i/(i+1.)) + d_loss.item()*(1./(i+1.))
            g_losses[epoch] = g_losses[epoch]*(i/(i+1.)) + g_loss.item()*(1./(i+1.))
            real_scores[epoch] = real_scores[epoch]*(i/(i+1.)) + real_score.mean().item()*(1./(i+1.))
            fake_scores[epoch] = fake_scores[epoch]*(i/(i+1.)) + fake_score.mean().item()*(1./(i+1.))
            
            if (i+1) % 200 == 0:
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                    .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                            real_score.mean().item(), fake_score.mean().item()))



if __name__ == '__main__':
    opt = parse_option()

    generator, discriminator, optimizer_G, optimizer_D, adversarial_loss, dataloader = build_model(opt)

    train(opt)