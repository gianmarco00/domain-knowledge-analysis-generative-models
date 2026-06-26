import torch
from torch.utils.data import DataLoader, TensorDataset

from domain_knowledge_analysis.models import Vae
from domain_knowledge_analysis.training import Trainer
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

image_shape = [image_channels, 28, 28]

lr = 1e-3

epochs = 20

device = "cuda" if torch.cuda.is_available() else "mps"

def test_fit_output(batch_size=batch_size, image_shape=image_shape, encoder_params=encoder_params, lr=lr, epochs=epochs, device=device):
    
    images = torch.rand(batch_size, *image_shape)
    dataset = TensorDataset(images)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = Vae(image_shape=image_shape, encoder_params=encoder_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(model, dataloader, dataloader, optimizer, vae_loss, epochs, device)

    history = trainer.fit()

    assert len(history["train_loss"]) == epochs
    assert len(history["validation_loss"]) == epochs

def test_train_epoch_parameter_change(batch_size=batch_size, image_shape=image_shape, encoder_params=encoder_params, lr=lr, epochs=epochs, device=device):
    
    images = torch.rand(batch_size, *image_shape)
    dataset = TensorDataset(images)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = Vae(image_shape=image_shape, encoder_params=encoder_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(model, dataloader, dataloader, optimizer, vae_loss, epochs, device)

    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

    trainer.train_epoch()

    parameters_after = [parameter.detach().clone() for parameter in model.parameters()]

    has_changed = any(
        not torch.allclose(before, after)
        for before, after in zip(parameters_before, parameters_after)
    )
    assert has_changed, "Expected model parameters to change after training, but they did not."

def test_validate_epoch_parameter_change(batch_size=batch_size, image_shape=image_shape, encoder_params=encoder_params, lr=lr, epochs=epochs, device=device):
    
    images = torch.rand(batch_size, *image_shape)
    dataset = TensorDataset(images)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = Vae(image_shape=image_shape, encoder_params=encoder_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(model, dataloader, dataloader, optimizer, vae_loss, epochs, device)

    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

    trainer.validate_epoch()

    parameters_after = [parameter.detach().clone() for parameter in model.parameters()]

    has_changed = any(
        not torch.allclose(before, after)
        for before, after in zip(parameters_before, parameters_after)
    )
    assert not has_changed, "Expected model parameters to stay the same for validation, but they changed."

