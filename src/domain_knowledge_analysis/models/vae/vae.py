import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

from domain_knowledge_analysis.math.gaussian import sample_gaussian

class Vae(nn.Module):
    def __init__(self, image_shape, encoder_params, decoder_params=None):
        super(Vae, self).__init__()

        self.image_shape = image_shape
        self.encoder_params = encoder_params

        self.encoder = Encoder(self.image_shape, encoder_params)

        if decoder_params is None:
            decoder_params = self.derive_decoder_params_from_encoder(self.encoder, encoder_params) 

        self.decoder = Decoder(decoder_params)

    def forward(self, x):
        mean, log_variance = self.encoder(x)
        z = self.reparametrize(mean, log_variance)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, log_variance
    
    def reparametrize(self, mean, log_variance):
        std = torch.exp(0.5*log_variance)
        eps = torch.randn_like(std)
        return mean + eps*std
    
    def generate_images(self, n_images):

        device = next(self.parameters()).device
       
        z = torch.randn(n_images, self.encoder_params["latent_dim"]).to(device)
        logits = self.decoder(z)
        x = torch.sigmoid(logits)

        return x
    
    def reconstruct_images(self, x):
        device = next(self.parameters()).device
        x = x.to(device)

        mean, log_variance = self.encoder(x)
        logits = self.decoder(mean)
        reconstructed_x = torch.sigmoid(logits)

        return reconstructed_x


    @staticmethod
    def derive_decoder_params_from_encoder(encoder, encoder_params):
        encoder_shapes = list(encoder.intermediate_shapes)
        decoder_shapes = list(reversed(encoder_shapes))

        decoder_kernels = list(reversed(encoder_params["kernels"]))
        decoder_strides = list(reversed(encoder_params["strides"]))
        decoder_paddings = list(reversed(encoder_params["paddings"]))

        encoder_channels = [
            encoder.image_shape[0],
            *encoder_params["out_channels"],
        ]

        decoder_channels = list(reversed(encoder_channels))

        decoder_out_channels = decoder_channels[1:]

        decoder_in_channels = decoder_channels[:-1]

        output_paddings = []

        for layer_idx, (current_shape, target_shape, k, s, p) in enumerate(
            zip(
                decoder_shapes[:-1],
                decoder_shapes[1:],
                decoder_kernels,
                decoder_strides,
                decoder_paddings,
            )
        ):
            current_H, current_W = current_shape[1], current_shape[2]
            target_H, target_W = target_shape[1], target_shape[2]

            base_H = (current_H - 1) * s - 2 * p + k
            base_W = (current_W - 1) * s - 2 * p + k

            output_padding_H = target_H - base_H
            output_padding_W = target_W - base_W

            if output_padding_H != output_padding_W:
                output_padding = (output_padding_H, output_padding_W)
            else:
                output_padding = output_padding_H

            if isinstance(output_padding, int):
                assert 0 <= output_padding < s, (
                    f"Invalid output_padding={output_padding} "
                    f"for decoder layer {layer_idx} with stride={s}"
                )
            else:
                assert 0 <= output_padding[0] < s
                assert 0 <= output_padding[1] < s

            output_paddings.append(output_padding)

        decoder_params = {
            "latent_dim": encoder_params["latent_dim"],
            "fc_neurons": encoder.fc_neurons,
            "last_shape_before_flattening": encoder.last_shape_before_flattening,
            "out_channels": decoder_out_channels,
            "in_channels": decoder_in_channels,
            "kernels": decoder_kernels,
            "strides": decoder_strides,
            "paddings": decoder_paddings,
            "output_paddings": output_paddings,
            "decoder_shapes": decoder_shapes,
        }

        return decoder_params

