import torch
from domain_knowledge_analysis.models.vae import Encoder
from domain_knowledge_analysis.models.vae import Decoder
from domain_knowledge_analysis.models.vae import Vae
from domain_knowledge_analysis.losses import vae_loss

image_channels = 3
batch_size = 4

encoder_params = {
    "latent_dim": 12,
    "out_channels": [24, 24, 24, 24],
    "kernels": [3, 3, 3, 3],
    "strides": [1, 2, 2, 1],
    "paddings": [1, 1, 1, 1],
    "in_channels": image_channels,
}

encoder_params = {
    "latent_dim": 32,
    "out_channels": [32, 64, 128, 128],
    "kernels": [3, 3, 3, 3],
    "strides": [1, 2, 2, 1],
    "paddings": [1, 1, 1, 1],
}

image_shape = [image_channels, 28, 28]

def test_encoder_returns_correct_shapes(batch_size=batch_size, image_shape=image_shape, encoder_params=encoder_params):

    encoder = Encoder(image_shape, encoder_params)

    x = torch.randn(batch_size, *image_shape)

    mean, log_variance = encoder(x)

    assert mean.shape == torch.Size((batch_size, encoder_params["latent_dim"])), f"Expected mean shape {(batch_size, encoder_params['latent_dim'])}, but got {mean.shape}"
    assert log_variance.shape == torch.Size((batch_size, encoder_params["latent_dim"])), f"Expected log_variance shape {(batch_size, encoder_params['latent_dim'])}, but got {log_variance.shape}"

def test_decoder_returns_correct_shapes(batch_size=batch_size, image_shape=image_shape, encoder_params=encoder_params):
    
    encoder = Encoder(image_shape, encoder_params)

    x = torch.randn(batch_size, *image_shape)

    vae = Vae(image_shape, encoder_params)
    decoder = Decoder(vae.derive_decoder_params_from_encoder(encoder, encoder_params))

    mean, log_variance = encoder(x)

    z = torch.randn(mean.shape)  

    result = decoder(z)

    assert result.shape == torch.Size((batch_size, *image_shape)), f"Expected output shape {x.shape}, but got {result.shape}"

def test_decoder_mirrors_encoder(batch_size=batch_size, image_shape=image_shape, encoder_params=encoder_params):
    
    encoder = Encoder(image_shape, encoder_params)
    encoder_shapes = list(encoder.intermediate_shapes)

    vae = Vae(image_shape, encoder_params)
    decoder_params = vae.derive_decoder_params_from_encoder(encoder, encoder_params)
    decoder = Decoder(decoder_params)
    decoder_shapes = list(decoder_params["decoder_shapes"])

    print("\nCalculated decoder params:")
    for key, value in decoder_params.items():
        if key not in ["encoder_shapes", "decoder_shapes"]:
            print(f"{key}: {value}")

    assert encoder_shapes == list(reversed(decoder_shapes)), f"Expected decoder shapes to mirror encoder shapes, but got\n {decoder_shapes}\n and\n {encoder_shapes}"

def test_vae_returns_correct_shapes(batch_size=batch_size, image_shape=image_shape , encoder_params=encoder_params):

    x = torch.randn(batch_size, *image_shape)

    vae = Vae(image_shape, encoder_params)

    reconstructed_x, mean, log_variance = vae(x)

    assert reconstructed_x.shape == x.shape
    assert mean.shape == (batch_size, encoder_params["latent_dim"])
    assert log_variance.shape == (batch_size, encoder_params["latent_dim"])

def test_vae_reparametrization(batch_size=batch_size, image_shape=image_shape, encoder_params=encoder_params):

    vae = Vae(image_shape, encoder_params)

    mean = torch.randn(batch_size, encoder_params["latent_dim"])

    large_negative_value = -10000

    log_variance = torch.full((batch_size, encoder_params["latent_dim"]), large_negative_value)

    z = vae.reparametrize(mean, log_variance)

    assert torch.allclose(z, mean), "Reparametrization did not return the mean when log_variance is very negative."

def test_vae_loss(batch_size=batch_size, image_shape=image_shape, encoder_params=encoder_params):

    x = torch.rand(batch_size, *image_shape)

    vae = Vae(image_shape, encoder_params)

    reconstructed_x, mean, log_variance = vae(x)

    loss = vae_loss(x, reconstructed_x, mean, log_variance)

    assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor"
    assert loss.shape == torch.Size([]), f"Loss should be a scalar tensor and instead is of shape {loss.shape}"

def test_generate_images(image_shape=image_shape, encoder_params=encoder_params):

    vae = Vae(image_shape, encoder_params)

    number_of_images = 5

    x = vae.generate_images(number_of_images)

    assert x.shape == torch.Size([number_of_images, *image_shape])