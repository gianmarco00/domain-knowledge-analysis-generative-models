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

        self.num_images_to_log = 36
        self.num_images_to_reconstruct = 16
        self.reconstruction_samples = next(iter(validate_dataloader))[0][:self.num_images_to_reconstruct].detach().cpu()

        self.best_validation_loss = float("inf")

        self.model.to(self.device)

        if start_weights != "random" and checkpoint_manager is None:
            raise ValueError("Cannot load start_weights without a checkpoint_manager.")

        if start_weights != "random":
            self.model = checkpoint_manager.load_model(self.model, start_weights, self.device, self.optimizer)

        self.history = {
            "train_loss": [],
            "validation_loss": [],
        }
    
    def train_epoch(self):

        self.model.train()
        total_loss = 0
        total_samples = 0

        for batch in self.train_dataloader:

            x = batch[0].to(self.device)
            self.optimizer.zero_grad()
            logits, mean, log_variance = self.model(x)
            loss_value = self.loss(x, logits, mean, log_variance)
            loss_value.backward()
            self.optimizer.step()

            total_loss += loss_value.item() * len(x)
            total_samples += len(x)

        return total_loss / total_samples
    
    def validate_epoch(self):

        self.model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.validate_dataloader:

                x = batch[0].to(self.device)
                logits, mean, log_variance = self.model(x)
                loss_value = self.loss(x, logits, mean, log_variance)

                total_loss += loss_value.item() * len(x)
                total_samples += len(x)

        return total_loss / total_samples
    
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
            
            if self.checkpoint_manager is not None:
                if validation_loss < self.best_validation_loss:
                    self.best_validation_loss = validation_loss
                    self.checkpoint_manager.save_best(self.model, self.optimizer, epoch+1, self.history, validation_loss)

            if self.logger is not None:

                self.logger.log_scalar("Loss/train", train_loss, epoch+1)
                self.logger.log_scalar("Loss/validation", validation_loss, epoch+1)

                self.generate_and_log_random_images("Images/Random", epoch)
                self.generate_and_log_reconstructed_images("Images/Reconstructed Images", epoch)

                if (epoch+1) % 10 == 0:
                    self.generate_and_log_random_images("Images/Random", epoch+1)
                    self.generate_and_log_reconstructed_images("Images/Reconstructed Images", epoch+1)

                self.logger.flush()

        if self.checkpoint_manager is not None:
            self.checkpoint_manager.save_last(self.model, self.optimizer, self.epochs, self.history, validation_loss)

        if self.logger is not None:
            self.logger.close()

        return self.history

    def generate_and_log_random_images(self, name, epoch):
        
        self.model.eval()
        with torch.no_grad():
            images = self.model.generate_images(self.num_images_to_log)
            self.logger.log_images(name, images, epoch)

    def generate_and_log_reconstructed_images(self, name, epoch):

        self.model.eval()
        with torch.no_grad():
            reconstructed_samples = self.model.reconstruct_images(self.reconstruction_samples)
            self.logger.log_image_pairs(name, self.reconstruction_samples, reconstructed_samples, epoch)