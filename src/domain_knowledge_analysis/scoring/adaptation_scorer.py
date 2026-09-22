import torch

class AdaptationScorer:
    def __init__(self, model, lora_manager, target_dataloader, source_dataloader, config, device):
        self.model = model
        self.lora_manager = lora_manager

        self.target_dataloader = target_dataloader
        self.source_dataloader = source_dataloader

        self.config = config
        self.device = device

    def score(self):
        self.lora_manager.enable_adapters()
        adapted_target_mse = self.score_mse(self.model, self.target_dataloader)
        adapted_source_mse = self.score_mse(self.model, self.source_dataloader)

        self.lora_manager.disable_adapters() 
        pretrained_target_mse = self.score_mse(self.model, self.target_dataloader)
        pretrained_source_mse = self.score_mse(self.model, self.source_dataloader)

        return {
            "adapted_target_mse": adapted_target_mse,
            "adapted_source_mse": adapted_source_mse,
            "pretrained_target_mse": pretrained_target_mse,
            "pretrained_source_mse": pretrained_source_mse,
        }

    def score_mse(self, model, dataloader):
        model.eval()
        total_mse = 0.0

        with torch.no_grad():
            
            for batch in dataloader:

                x = batch[0].to(self.device)
                outputs = model(x)

                mse = torch.nn.functional.mse_loss(outputs, x)
                mse.flatten(start_dim=1).mean(dim=1)

                total_mse = mse.sum()

        mean_mse = total_mse / len(dataloader.dataset)

        return mean_mse.item()
