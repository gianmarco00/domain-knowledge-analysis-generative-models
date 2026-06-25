import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, params):
        super(Encoder, self).__init__()
        
        layers = []

        kernels = params.get('kernels')
        out_channels = params.get('out_channels')
        strides = params.get('strides')
        paddings = params.get('paddings')

        in_channels = [input_dim[1], *out_channels[:-1]]

        for in_ch, out_ch, k, s, p in zip(in_channels, out_channels, kernels, strides, paddings):

            layers.append(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p))
            layers.append(nn.ReLU())

        self.conv = nn.Sequential(*layers)

        self.flatten = nn.Flatten(start_dim=1)

        # Compute the shape of the output after convolution
        with torch.no_grad():

            dummy = torch.zeros(input_dim)
            result = self.conv(dummy)
            final_shape = result.shape[:-1]  # Exclude the batch dimension

        self.fc_neurons = final_shape.numel()

        self.fc_mean = nn.Linear(self.fc_neurons, params.get('latent_dim')) 
        self.fc_variance = nn.Linear(self.fc_neurons, params.get('latent_dim')) 

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        mean = self.fc_mean(x)
        log_variance = self.fc_variance(x)
        return mean, log_variance
        