

import os
import yaml
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import ImageDataset
from models import VAE, VAE_CNN
from ml_functions import train, train_vae, validate_vae

import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import classification_report

from utils import plot_latent_space


def run_ml(config_file = ''):
  
    # Initialize Parameters

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    for key in config:
        setattr(args, key, config[key])

    main_dir = os.path.join('anomaly_dataset',args.dataset)
    train_dir = os.path.join(main_dir,'train')

    training_data = ImageDataset.ImageDataset(
        img_dir = train_dir,
        with_canny=args.with_canny,
        transform=ToTensor()
    )
    train_loader = DataLoader(training_data,batch_size=args.batch_size,shuffle=True)

    # Train Model

    if args.model == 'VAE':
        model = VAE(input_dim = 256*256, hidden_dim = 256, hidden_dim2 =16)
        criterion = nn.CrossEntropyLoss()
    elif args.model == 'VAE_CNN':
        model = VAE_CNN()

    optimizer = torch.optim.Adam(model.parameters(),
                                args.learning_rate)
    
    for epoch in range(args.epochs):

        # train loop
        if 'VAE' in model.__class__.__name__:
            train_vae(epoch, train_loader, model, optimizer)
        else:
            train(epoch, train_loader, model, optimizer, criterion)

        if (epoch % args.save_checkpoint) == 0:

            torch.save(model,'./models/' + args.savename + '_' + str(epoch) + '.pth')

    torch.save(model,'./models/' + args.savename + '.pth')


def evaluate_models(config_file = ''):

    # Initialize Parameters

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    for key in config:
        setattr(args, key, config[key])

    main_dir = os.path.join('anomaly_dataset',args.dataset)

    train_dir = os.path.join(main_dir,'train')
    test_dir = os.path.join(main_dir,'test')

    # Model Evaluation

    model = torch.load('./models/' + args.savename + '.pth', weights_only=False)
    model.eval()
    
    training_data = ImageDataset.ImageDataset(
        img_dir = train_dir,
        with_canny=args.with_canny,
        transform=ToTensor()
    )
    
    #set batch_size = 1 to evaluate loss of each sample
    train_loader = DataLoader(training_data,batch_size=1,shuffle=True)

    test_data = ImageDataset.ImageDataset(
        img_dir = test_dir,
        with_canny=args.with_canny,
        transform=ToTensor()
    )
    test_loader = DataLoader(test_data,batch_size=1,shuffle=True)

    # Calculate loss for each training sample to define a classification threshold
    mean, log_var, loss, label = validate_vae(0, train_loader, model)
    loss_threshold = np.percentile(loss,90)
    
    # Perform model inference on each testing sample
    mean2, log_var2, loss2, label2 = validate_vae(0, test_loader, model)

    # Classify testing samples based on loss
    y_pred = loss2>loss_threshold # 1 if predicted as anomoly, 0 if predicted as good image
    y_true = label2>=1 # 1 if true anomaly, 0 if true good image
    
    print(classification_report(y_true,y_pred,labels = [0,1],target_names=['Good','Anomaly']))

    # Set to True to produce supplemental plots
    if True:
        
        plot_latent_space(model, scale=1.0, n=5, digit_size=28, figsize=15, savename=args.savename)
        
        plt.figure()
        plt.scatter(mean[:,0],mean[:,1],c=label)
        plt.scatter(mean2[:,0],mean2[:,1],c=label2,marker='.')
        plt.savefig('plots/' + args.savename + '_mean.png')

        plt.figure()
        plt.scatter(log_var[:,0],log_var[:,1],c=label)
        plt.scatter(log_var2[:,0],log_var2[:,1],c=label2,marker='.')
        plt.savefig('plots/' + args.savename + '_var.png')

        plt.figure()
        plt.scatter(mean[:,0],log_var[:,0],c=label)
        plt.scatter(mean2[:,0],log_var2[:,0],c=label2,marker='.')
        plt.savefig('plots/' + args.savename + '_mean_var0.png')

        plt.figure()
        plt.scatter(mean[:,1],log_var[:,1],c=label)
        plt.scatter(mean2[:,1],log_var2[:,1],c=label2,marker='.')
        plt.savefig('plots/' + args.savename + '_mean_var1.png')

        plt.figure()
        plt.scatter(range(len(loss)),loss,c=label)
        plt.scatter(range(len(loss2)),loss2,c=label2,marker='.')
        plt.savefig('plots/' + args.savename + '_loss.png')


if __name__ == '__main__':
    
    # Set to True to train and save models
    if True:

        config_list = ['vae_raw','vae_canny','vae_cnn_raw','vae_cnn_canny']

        for config in config_list:

            print('STARTING TRAINING:' + config)
            run_ml('configs/' + config + '.yaml')

            
    # Set to True to evaluate models and classify testing images
    if True:

        config_list = ['vae_raw','vae_canny','vae_cnn_raw','vae_cnn_canny']

        for config in config_list:
            
            print('STARTING EVAL:' + config)
            evaluate_models('configs/' + config + '.yaml')