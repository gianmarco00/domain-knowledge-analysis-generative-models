import torch

class AdaptationScorer:
    def __init__(self, model, lora_manager, target_dataloader, source_dataloader, config, device):
        self.model = model
        self.lora_manager = lora_manager

        self.target_dataloader = target_dataloader
        self.source_dataloader = source_dataloader

        self.config = config
        self.device = device

        self.num_images = 5
        self.original_target_images = next(iter(self.target_dataloader))[0][:self.num_images]
        self.original_source_images = next(iter(self.source_dataloader))[0][:self.num_images]

    def score(self):
        self.lora_manager.enable_adapters()
        adapted_target_mse = self.score_mse(self.model, self.target_dataloader)
        adapted_source_mse = self.score_mse(self.model, self.source_dataloader)
        adapted_reconstructed_target_images = self.reconstruct_images(self.model, self.original_target_images)
        adapted_reconstructed_source_images = self.reconstruct_images(self.model, self.original_source_images)

        self.lora_manager.disable_adapters() 
        pretrained_target_mse = self.score_mse(self.model, self.target_dataloader)
        pretrained_source_mse = self.score_mse(self.model, self.source_dataloader)
        pretrained_reconstructed_target_images = self.reconstruct_images(self.model, self.original_target_images)
        pretrained_reconstructed_source_images = self.reconstruct_images(self.model, self.original_source_images)

        return {
            "adapted_target_mse": adapted_target_mse,
            "adapted_source_mse": adapted_source_mse,
            "pretrained_target_mse": pretrained_target_mse,
            "pretrained_source_mse": pretrained_source_mse,
            "original_target_images": self.original_target_images,
            "original_source_images": self.original_source_images,
            "adapted_reconstructed_target_images": adapted_reconstructed_target_images,
            "adapted_reconstructed_source_images": adapted_reconstructed_source_images,
            "pretrained_reconstructed_target_images": pretrained_reconstructed_target_images,
            "pretrained_reconstructed_source_images": pretrained_reconstructed_source_images,
        }

    def score_mse(self, model, dataloader):
        model.eval()
        total_mse = 0.0

        with torch.no_grad():
            
            for batch in dataloader:

                x = batch[0].to(self.device)
                outputs, _, _ = model(x)

                mse = torch.nn.functional.mse_loss(outputs, x)
                mse.flatten(start_dim=1).mean(dim=1)

                total_mse += mse.sum()

        mean_mse = total_mse / len(dataloader.dataset)

        return mean_mse.item()
    
    def reconstruct_images(self, model, images):
        model.eval()
        reconstructions = []

        with torch.no_grad():
            for x in images:

                x = x.to(self.device)
                outputs = model(x)

                reconstructions.append(outputs.cpu())

        return reconstructions
