import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, train_dataloader, validate_dataloader, optimizer, loss, epochs, device, logger=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.device = device
        self.logger = logger

        self.model.to(self.device)

        

        self.history = {
            "train_loss": [],
            "validation_loss": [],
        }
    
    def train_epoch(self):

        self.model.train()
        total_loss = 0

        for batch in self.train_dataloader:

            x = batch[0].to(self.device)
            self.optimizer.zero_grad()
            logits, mean, log_variance = self.model(x)
            loss_value = self.loss(x, logits, mean, log_variance)
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
                logits, mean, log_variance = self.model(x)
                loss_value = self.loss(x, logits, mean, log_variance)

                total_loss += loss_value.item()

        return total_loss / len(self.validate_dataloader)
    
    def fit(self):

        progress_bar = tqdm(range(self.epochs), desc="Training")

        for epoch in progress_bar:

            train_loss = self.train_epoch()
            validation_loss = self.validate_epoch()

            self.history["train_loss"].append(train_loss)
            self.history["validation_loss"].append(validation_loss)

            progress_bar.set_postfix({
                    "train_loss": train_loss,
                    "validation_loss": validation_loss
                })

            if self.logger is not None:

                self.logger.log_scalar("Loss/train", train_loss, epoch)
                self.logger.log_scalar("Loss/validation", validation_loss, epoch)
                self.logger.flush()

        if self.logger is not None:
            self.logger.close()

        return self.history
