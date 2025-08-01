

import numpy as np
from skimage import io
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def read_image(image_fname: str):

    '''read image and convert to float'''

    im = io.imread(image_fname)
    im = np.uint8(im)
    return im

def clean_plot(img, save_path = ''):

    '''concise function to plot images, with option to save'''

    fig = plt.figure()
    #plt.imshow(img)
    plt.imshow(img, cmap='gray',vmin=0,vmax=1)
    plt.axis('off')
    if len(save_path)>1:
        plt.savefig(save_path)
        plt.close()


def plot_latent_space(model, scale=1.0, n=25, digit_size=28, figsize=15, savename=''):

    '''Plot the latent space of trained VAE model.
    Original source - https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f'''

    # display a n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))

    # construct a grid 
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float)
            x_decoded = model.decode(z_sample)
            
            #digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            digit = x_decoded[0].detach().cpu().reshape(256, 256) #TODO - dynamically define shape
            digit = resize(digit.numpy(),(digit_size,digit_size))
            
            figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size,] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.title('VAE Latent Space Visualization')
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("mean, z [0]")
    plt.ylabel("var, z [1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig('plots/' + savename + '_latent_space.png')

