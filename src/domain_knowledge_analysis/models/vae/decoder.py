import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        layers = []

        kernels = params["kernels"]
        in_channels = params["in_channels"]
        out_channels = params["out_channels"]
        strides = params["strides"]
        paddings = params["paddings"]
        output_paddings = params["output_paddings"]
        fc_neurons = params["fc_neurons"]
        start_shape = params["last_shape_before_flattening"]

        self.fc = nn.Linear(params["latent_dim"], fc_neurons)
        self.fc_activation = nn.LeakyReLU()

        self.unflatten = nn.Unflatten(
            dim=1,
            unflattened_size=start_shape,
        )

        for n_layer, (in_ch, out_ch, k, s, p, op) in enumerate(
            zip(
                in_channels,
                out_channels,
                kernels,
                strides,
                paddings,
                output_paddings,
            )
        ):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    output_padding=op,
                )
            )

            if n_layer < len(out_channels) - 1:
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.LeakyReLU())

        self.conv_transpose = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        x = self.fc_activation(x)
        x = self.unflatten(x)
        x = self.conv_transpose(x)

        return x
        