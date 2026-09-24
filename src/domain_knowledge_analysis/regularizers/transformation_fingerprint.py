import torch

class TFRegularizer(torch.nn.Module):
    def __init__(self, model, lora_manager, transformations, anchor_points):
        super().__init__()
        self.model = model
        self.lora_manager = lora_manager
        self.transformations_functions, self.intensities = transformations
        self.num_transformations = len(self.transformations_functions)
        self.x = anchor_points
        
        self.calibrate()

    def calibrate(self):
        self.lora_manager.disable_adapters()
        with torch.no_grad():
            self.source_gram_matrix = self.gram_matrix(self.x)
        self.lora_manager.enable_adapters()

    def apply_transform(self, x):
        with torch.no_grad():

            transformed_x = torch.stack([t(x) for t in self.transformations_functions],dim=1)

        return transformed_x
    
    def model_response(self, x):
        transformed_x = self.apply_transform(x)
        transformed_output = self.model.deterministic_forward(transformed_x.flatten(0,1)).reshape(len(x), self.num_transformations, -1) 
        original_output = self.model.deterministic_forward(x).flatten(start_dim=1).unsqueeze(1)
        response = transformed_output - original_output
        response = response / self.intensities.view(1, -1, 1)

        return response
    
    def gram_matrix(self, x):
        response_vector = self.model_response(x)
        g_matrix = (response_vector @ response_vector.transpose(-1, -2)).mean(dim=0)
        trace = torch.diagonal(g_matrix).sum()
        return g_matrix / trace

    def forward(self):
        regularizer_loss = torch.square(self.gram_matrix(self.x) - self.source_gram_matrix).sum()
        return regularizer_loss
        