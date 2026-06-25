import torch
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, train_dataloader, validate_dataloader, optimizer, loss, epochs, device, log_dir=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.device = device

        self.model.to(self.device)

        self.writer = SummaryWriter(log_dir) if log_dir is not None else None
    
    def train_epoch(self):

        self.model.train()
        total_loss = 0

        for batch in self.train_dataloader:

            x = batch[0].to(self.device)
            self.optimizer.zero_grad()
            reconstructed_x, mean, log_variance = self.model(x)
            loss_value = self.loss(x, reconstructed_x, mean, log_variance)
            loss_value.backward()
            self.optimizer.step()

            total_loss += loss_value.item()

        return total_loss / len(self.train_dataloader)
    
    def validate_epoch(self):

        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.validate_dataloader:

                x = batch[0].to(self.device)
                reconstructed_x, mean, log_variance = self.model(x)
                loss_value = self.loss(x, reconstructed_x, mean, log_variance)

                total_loss += loss_value.item()

        return total_loss / len(self.validate_dataloader)
    
    def fit(self):

        for epoch in range(self.epochs):

            train_loss = self.train_epoch()
            validate_loss = self.validate_epoch()

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/validation", validate_loss, epoch)

        if self.writer is not None:
            self.writer.close() 

        return None
