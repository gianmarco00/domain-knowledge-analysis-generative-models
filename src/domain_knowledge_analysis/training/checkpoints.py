import torch

class CheckpointManager():
    def __init__(self, log_dir, config=None):

        self.log_dir = log_dir
        self.config = config

        self.checkpoint_dir = log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


    def checkpoint_path(self, filename):
        return self.checkpoint_dir / filename

    def save_checkpoint(self, model, optimizer, epoch, history, validation_loss, filename):

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
            "config": self.config,
            "validation_loss": validation_loss
        },
        self.checkpoint_path(filename)
        )
    
    def save_last(self, model, optimizer, epoch, history, validation_loss):
        self.save_checkpoint(model, optimizer, epoch, history, validation_loss, filename="last.pt")
    
    def save_best(self, model, optimizer, epoch, history, validation_loss):
        self.save_checkpoint(model, optimizer, epoch, history, validation_loss, filename="best.pt")


    def load_model(self, model, checkpoint_path, device, optimizer=None):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.model_config = checkpoint["config"]

        if self.config is not None:
            if self.model_config["model"] != self.config["model"]:
                raise ValueError(
                    "Pretrained model architecture differs from current model architecture.\n"
                    f"Checkpoint model config:\n{self.model_config['model']}\n"
                    f"Current model config:\n{self.config['model']}"
                )

        self.training_dataset = self.model_config["dataset"]["name"]

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return model
        

    
