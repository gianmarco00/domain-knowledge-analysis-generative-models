import torch

class CheckpointManager():
    def __init__(self, log_dir, config):

        self.log_dir = log_dir
        self.config = config

        self.checkpoint_dir = log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


    def checkpoint_path(self, filename):
        return self.checkpoint_dir / filename

    def save_checkpoint(self, model, optimizer, epoch, history, filename):

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
            "config": self.config
        },
        self.checkpoint_path(filename)
        )
    
    def save_last(self, model, optimizer, epoch, history):

        self.save_checkpoint(model, optimizer, epoch, history, filename="last.pt")

    def load_model(self, model, checkpoint_path, device, optimizer=None):

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        

    
