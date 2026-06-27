import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_dataloader, validate_dataloader, optimizer, loss, epochs, device, checkpoint_manager=None, logger=None, start_weights="random"):
        self.model = model
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.device = device
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager

        self.num_images_to_log = 8

        self.model.to(self.device)

        if start_weights != "random" and checkpoint_manager is None:
            raise ValueError("Cannot load start_weights without a checkpoint_manager.")

        if start_weights != "random":
            checkpoint_manager.load_model(model, start_weights, device, optimizer)

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

                self.logger.log_scalar("Loss/train", train_loss, epoch+1)
                self.logger.log_scalar("Loss/validation", validation_loss, epoch+1)

                if epoch % 10 == 0:
                    self.generate_and_log_images(f"Generated samples for epoch {epoch}", epoch+1)

                self.logger.flush()

        if self.checkpoint_manager is not None:
            self.checkpoint_manager.save_last(self.model, self.optimizer, self.epochs, self.history)

        if self.logger is not None:
            self.generate_and_log_images(f"Generated samples for last epoch", epoch+1)
            self.logger.close()

        return self.history

    def generate_and_log_images(self, name, epoch):
        
        self.model.eval()
        with torch.no_grad():
            images = self.model.generate_images(self.num_images_to_log)
            self.logger.log_images(name, images, epoch, self.num_images_to_log)
