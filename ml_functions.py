import time
from utils import AverageMeter, clean_plot
import torch

def train(epoch, data_loader, model, optimizer, criterion):

    '''Function to execute the model training process.
    Used by non-VAE models so this function is obsolete for this project.'''

    iter_time = AverageMeter()
    losses = AverageMeter()

    for idx, (data, target) in enumerate(data_loader):
        start = time.time()

        # forward pass - compute predicted y by passing x to the model
        out = model(data)
        loss = criterion(out, target)

        # Zero gradients, perform a backward pass, and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Training Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t').format(
                       epoch,
                       idx,
                       len(data_loader),
                       iter_time=iter_time,
                       loss=losses))


def vae_loss_function(x, x_hat, mean, log_var):

    '''Loss function for the VAE models.
    Original source - 
    https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f'''

    reproduction_loss = torch.nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


def train_vae(epoch, data_loader, model, optimizer):

    '''Function to execute the model training process for VAE architectures'''

    iter_time = AverageMeter()

    overall_loss = 0
    for idx, (data, target) in enumerate(data_loader):
        start = time.time()

        # forward pass - compute predicted y by passing x to the model
        if 'CNN' in model.__class__.__name__:
            x = data
        else:
            x = data.view(data.shape[0],data.shape[2]*data.shape[3]) 
        x_hat, mean, log_var = model(x)
        loss = vae_loss_function(x, x_hat, mean, log_var)

        overall_loss += loss.item()

        # Zero gradients, perform a backward pass, and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print('Training Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss:.4f} /{overall_loss:.4f}  \t'.format(
                       epoch,
                       idx,
                       len(data_loader),
                       iter_time=iter_time,
                       loss=loss,
                       overall_loss=overall_loss))
            

def validate_vae(epoch, val_loader, model):

    '''Function to execute the model inference process for VAE architectures'''

    iter_time = AverageMeter()

    mean_vec = torch.tensor([])
    log_var_vec = torch.tensor([])
    loss_vec = torch.tensor([])
    target_vec = torch.tensor([])
    for idx, (data, target) in enumerate(val_loader):
        start = time.time()

        model.eval()

        with torch.no_grad():

            if 'CNN' in model.__class__.__name__:
                x = data
            else:
                x = data.view(data.shape[0],data.shape[2]*data.shape[3]) 
            x_hat, mean, log_var = model(x)
            loss = vae_loss_function(x, x_hat, mean, log_var)
            
            mean_vec = torch.cat((mean_vec,mean),dim=0)
            log_var_vec = torch.cat((log_var_vec,log_var),dim=0)
            loss_vec = torch.cat((loss_vec,loss.unsqueeze(dim=0)),dim=0)
            target_vec = torch.cat((target_vec,target),dim=0)

            # Set to true during debugging to visually compare the original image and the decoded image
            if False:

                decoded_sample = x_hat[0].detach().cpu().reshape(256, 256)
                clean_plot(decoded_sample)

                original_image = data[0].detach().cpu().reshape(256, 256)
                clean_plot(original_image)
                
        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Validation Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f}\t').format(
                       epoch,
                       idx,
                       len(val_loader),
                       iter_time=iter_time))
    
    return mean_vec, log_var_vec, loss_vec, target_vec
