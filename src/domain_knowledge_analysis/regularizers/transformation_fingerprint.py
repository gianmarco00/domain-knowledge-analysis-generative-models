import torch

class TFRegularizer(torch.nn.Module):
    def __init__(self, model, lora_manager, transformations, anchor_points):
        super().__init__()
        self.model = model
        self.lora_manager = lora_manager
        self.transformations_functions, _ = transformations
        self.num_transformations = len(self.transformations_functions)
        self.x = anchor_points
        self.num_anchor_points = len(self.x)

        self.transformed_x = self.apply_transform()
        self.calibrate()

    def calibrate(self):

        input_delta = (self.transformed_x - self.x.unsqueeze(1)).flatten(start_dim=2)
        input_scales = torch.sqrt(torch.square(input_delta.mean(0,2)))
        self.input_scale_matrix = torch.outer(input_scales, input_scales)

        was_training = self.model.training
        self.model.eval()
        self.lora_manager.disable_adapters()
        with torch.no_grad():
            self.source_gram_matrix = self.gram_matrix()
        self.lora_manager.enable_adapters()
        if was_training:
            self.model.train()

    def apply_transform(self):
        with torch.no_grad():
            transformed_x = torch.stack([t(self.x) for t in self.transformations_functions], dim=1)

        return transformed_x
    
    def model_response(self):
        was_training = self.model.training
        self.model.eval()

        transformed_output = self.model.deterministic_forward(self.transformed_x.flatten(0,1)).reshape(self.num_anchor_points, self.num_transformations, -1) 
        original_output = self.model.deterministic_forward(self.x).flatten(start_dim=1).unsqueeze(1)
        response = transformed_output - original_output

        if was_training:
            self.model.train()

        return response
    
    def gram_matrix(self):
        response_vector = self.model_response()
        output_dimension = response_vector.shape[-1]
        g_matrix = (response_vector @ response_vector.transpose(-1, -2)).mean(dim=0) / output_dimension
        g_matrix = g_matrix / self.input_scale_matrix
        trace = torch.diagonal(g_matrix).sum()
        return g_matrix / trace

    def forward(self):
        regularizer_loss = torch.square(self.gram_matrix() - self.source_gram_matrix).sum()
        return regularizer_loss
        