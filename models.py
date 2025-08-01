
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2):

        """ Variational Autoencoder (Fully Connected Layers) architecture.
        input_dim: input feature dimension
        hidden_dim: hidden dimension
        hidden_dim2: second hidden dimension
        """

        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.LeakyReLU()
            )

        # latent mean and variance 
        self.mean_layer = nn.Linear(hidden_dim2, 2)
        self.logvar_layer = nn.Linear(hidden_dim2, 2)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, hidden_dim2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )
        
        self.double()

    # helper VAE functions from https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f

    def encode(self, x):
        out = self.encoder(x)
        mean, logvar = self.mean_layer(out), self.logvar_layer(out)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)  
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x.to(torch.double))

    def forward(self, x):
        #x = x.view(x.shape[0], -1)
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

class VAE_CNN(nn.Module):

    """ Variational Autoencoder (Convolutional) architecture."""

    def __init__(self):

        super(VAE_CNN, self).__init__()

        self.encoded_shape = (64, 24, 24)
        self.latent_dim = self.encoded_shape[0]*self.encoded_shape[1]*self.encoded_shape[2] # this must align with number of nodes in final FC layer

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 1,out_channels = 32,kernel_size = 9,stride=5), # output data size = (batch,32,50,50)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), # output data size = (batch,64,24,24)
            nn.ReLU()
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(self.latent_dim, 2)
        self.logvar_layer = nn.Linear(self.latent_dim, 2)

        # decoder
        self.decoder_fc = nn.Linear(2, self.latent_dim)
        self.relu = nn.ReLU()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=2,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32,out_channels=1,kernel_size=9,stride=5,output_padding=2),
            nn.Sigmoid()
            )

        self.double()

    def encode(self, x):
        out = self.encoder(x)
        out = out.view(out.shape[0],-1)

        mean, logvar = self.mean_layer(out), self.logvar_layer(out)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)  
        z = mean + var*epsilon
        return z

    def decode(self, x):
        x = self.decoder_fc(x.to(torch.double))
        x = self.relu(x)
        x = x.view(-1,*self.encoded_shape) 
        return self.decoder(x)

    def forward(self, x):
        #x = x.view(x.shape[0], -1)
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

