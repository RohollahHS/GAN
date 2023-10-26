import argparse
import numpy as np
from model import build_model
import matplotlib.pyplot as plt
import pylab
import numpy as np
from torchvision.utils import save_image
import os
from torchvision.utils import save_image

from torch.autograd import Variable
import torch


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_size", type=int, default=64, help="dimensionality of the latent space")
    parser.add_argument("--image_size", type=int, default=784, help="size of each image dimension")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_dir", default='samples')
    parser.add_argument("--save_dir", default='save')

    opt = parser.parse_args()

    if not os.path.exists(opt.sample_dir):
        os.makedirs(opt.sample_dir)

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    print('Args:')
    for k, v in vars(opt).items():
        print(f'{k}: {v}')
    print()

    return opt


def train(opt, G, D, g_optimizer, d_optimizer, criterion, data_loader):
    device = opt.device
    batch_size = opt.batch_size
    num_epochs = opt.n_epochs
    latent_size = opt.latent_size
    save_dir = opt.output_path
    sample_dir = opt.output_path

    d_losses = np.zeros(num_epochs)
    g_losses = np.zeros(num_epochs)
    real_scores = np.zeros(num_epochs)
    fake_scores = np.zeros(num_epochs)

    def reset_grad():
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

    total_step = len(data_loader)
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(data_loader):
            images = Variable(images.to(device))

            # Create the labels which are later used as input for the BCE loss
            real_labels = torch.ones(batch_size, 1).to(device)
            real_labels = Variable(real_labels)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            fake_labels = Variable(fake_labels)

            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #

            # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
            # Second term of the loss is always zero since real_labels == 1
            outputs = D(images)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs
            
            # Compute BCELoss using fake images
            # First term of the loss is always zero since fake_labels == 0
            z = torch.randn(batch_size, latent_size).to(device)
            z = Variable(z)
            fake_images = G(z)
            outputs = D(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs
            
            # Backprop and optimize
            # If D is trained so well, then don't update
            d_loss = d_loss_real + d_loss_fake
            reset_grad()
            d_loss.backward()
            d_optimizer.step()
            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #

            # Compute loss with fake images
            z = torch.randn(batch_size, latent_size).to(device)
            z = Variable(z)
            fake_images = G(z)
            outputs = D(fake_images)
            
            # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
            # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
            g_loss = criterion(outputs, real_labels)
            
            # Backprop and optimize
            # if G is trained so well, then don't update
            reset_grad()
            g_loss.backward()
            g_optimizer.step()
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
            
        # Save real images
        if (epoch+1) == 1:
            images = images.view(images.size(0), 1, 28, 28)
            save_image(denorm(images.data), os.path.join(sample_dir, 'real_images.png'))
        
        # Save sampled images
        fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
        save_image(denorm(fake_images.data), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))
        
        # Save and plot Statistics
        np.save(os.path.join(save_dir, 'd_losses.npy'), d_losses)
        np.save(os.path.join(save_dir, 'g_losses.npy'), g_losses)
        np.save(os.path.join(save_dir, 'fake_scores.npy'), fake_scores)
        np.save(os.path.join(save_dir, 'real_scores.npy'), real_scores)
        
        plt.figure()
        pylab.xlim(0, num_epochs + 1)
        plt.plot(range(1, num_epochs + 1), d_losses, label='d loss')
        plt.plot(range(1, num_epochs + 1), g_losses, label='g loss')    
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'loss.pdf'))
        plt.close()

        plt.figure()
        pylab.xlim(0, num_epochs + 1)
        pylab.ylim(0, 1)
        plt.plot(range(1, num_epochs + 1), fake_scores, label='fake score')
        plt.plot(range(1, num_epochs + 1), real_scores, label='real score')    
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'accuracy.pdf'))
        plt.close()

        # Save model at checkpoints
        if (epoch+1) % 50 == 0:
            torch.save(G.state_dict(), os.path.join(save_dir, 'G--{}.ckpt'.format(epoch+1)))
            torch.save(D.state_dict(), os.path.join(save_dir, 'D--{}.ckpt'.format(epoch+1)))

    # Save the model checkpoints 
    torch.save(G.state_dict(), 'G.ckpt')
    torch.save(D.state_dict(), 'D.ckpt')

if __name__ == '__main__':
    opt = parse_option()

    G, D, g_optimizer, d_optimizer, criterion, data_loader = build_model(opt)

    train(opt, G, D, g_optimizer, d_optimizer, criterion, data_loader)