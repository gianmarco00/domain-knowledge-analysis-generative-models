import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, image_shape, params):
        super(Encoder, self).__init__()

        self.image_shape = image_shape

        layers = []

        kernels = params["kernels"]
        out_channels = params["out_channels"]
        strides = params["strides"]
        paddings = params["paddings"]
        latent_dim = params["latent_dim"]

        in_channels = [image_shape[0], *out_channels[:-1]]

        for in_ch, out_ch, k, s, p in zip(
            in_channels, out_channels, kernels, strides, paddings
        ):
            layers.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                )
            )
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU())

        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten(start_dim=1)

        # Store intermediate shapes using dummy pass
        was_training = self.conv.training
        self.conv.eval()
        with torch.no_grad():
            dummy = torch.zeros(image_shape).unsqueeze(0)  

            self.intermediate_shapes = []
            self.intermediate_shapes.append(dummy.shape[1:])

            for layer in self.conv:
                dummy = layer(dummy)

                if isinstance(layer, nn.Conv2d):
                    self.intermediate_shapes.append(dummy.shape[1:])

            self.last_shape_before_flattening = self.intermediate_shapes[-1]
            self.fc_neurons = dummy.flatten(start_dim=1).shape[1]
        self.conv.train(was_training)

        self.fc_mean = nn.Linear(self.fc_neurons, latent_dim)
        self.fc_log_variance = nn.Linear(self.fc_neurons, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)

        mean = self.fc_mean(x)
        log_variance = self.fc_log_variance(x)

        return mean, log_variance