import torch
import torch.nn as nn


class AsymmetricDecoder(nn.Module):
    """
    A modestly wider asymmetric decoder for the current MNIST encoder.

    Expected encoder/latent configuration:
        latent dimension: configurable
        encoder output:   [batch, 128, 4, 4]

    Decoder output:
        logits: [batch, 1, 28, 28]

    The decoder returns logits. Do not apply sigmoid here.
    """

    def __init__(
        self,
        latent_dim,
        output_channels=1,
        use_batch_norm=True,
    ):
        super().__init__()

        self.start_channels = 256
        self.start_height = 4
        self.start_width = 4

        self.fc = nn.Linear(
            latent_dim,
            self.start_channels * self.start_height * self.start_width,
        )
        self.fc_activation = nn.LeakyReLU()

        self.unflatten = nn.Unflatten(
            dim=1,
            unflattened_size=(
                self.start_channels,
                self.start_height,
                self.start_width,
            ),
        )

        intermediate_bias = not use_batch_norm

        layers = [
            # 256 × 4 × 4 -> 128 × 4 × 4
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                output_padding=0,
                bias=intermediate_bias,
            ),
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(128))

        layers.append(nn.LeakyReLU())

        layers.append(
            # 128 × 4 × 4 -> 64 × 7 × 7
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
                bias=intermediate_bias,
            )
        )

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(64))

        layers.append(nn.LeakyReLU())

        layers.append(
            # 64 × 7 × 7 -> 32 × 14 × 14
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=intermediate_bias,
            )
        )

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(32))

        layers.append(nn.LeakyReLU())

        layers.append(
            # 32 × 14 × 14 -> output_channels × 28 × 28
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=True,
            )
        )

        self.conv_transpose = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z)
        x = self.fc_activation(x)
        x = self.unflatten(x)

        logits = self.conv_transpose(x)

        return logits