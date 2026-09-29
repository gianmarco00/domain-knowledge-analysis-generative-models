import torch


class TFRegularizer(torch.nn.Module):
    def __init__(self, model, lora_manager, transformations, anchor_points, anchor_batch_size=10):
        super().__init__()

        self.model = model
        self.lora_manager = lora_manager
        self.transformations_functions, _ = transformations
        self.num_transformations = len(self.transformations_functions)
        self.x = anchor_points
        self.num_anchor_points = len(self.x)
        self.anchor_batch_size = anchor_batch_size

        self.transformed_x = self.apply_transform()

        self.shuffle_anchor_indices()
        self.calibrate()

    def calibrate(self):
        input_delta = (self.transformed_x - self.x.unsqueeze(1)).flatten(start_dim=2)
        input_scales = input_delta.square().mean(dim=(0, 2)).clamp_min(1e-12).sqrt()
        self.input_scale_matrix = torch.outer(input_scales, input_scales)

        was_training = self.model.training
        self.model.eval()
        self.lora_manager.disable_adapters()

        with torch.no_grad():
            self.source_gram_matrices = self.gram_matrix()

        self.lora_manager.enable_adapters()

        if was_training:
            self.model.train()

    def shuffle_anchor_indices(self):
        self.anchor_indices = torch.randperm(self.num_anchor_points, device=self.x.device)
        self.anchor_position = 0

    def sample_anchor_indices(self):
        start = self.anchor_position
        end = min(start + self.anchor_batch_size, self.num_anchor_points)

        anchor_indices = self.anchor_indices[start:end]
        self.anchor_position = end

        if self.anchor_position == self.num_anchor_points:
            self.shuffle_anchor_indices()

        return anchor_indices

    def apply_transform(self):
        with torch.no_grad():
            transformed_x = torch.stack([t(self.x) for t in self.transformations_functions], dim=1)

        return transformed_x

    def model_response(self, anchor_indices=None):
        if anchor_indices is None:
            x = self.x
            transformed_x = self.transformed_x
        else:
            x = self.x[anchor_indices]
            transformed_x = self.transformed_x[anchor_indices]

        num_anchor_points = len(x)

        was_training = self.model.training
        self.model.eval()

        transformed_output = self.model.deterministic_forward(transformed_x.flatten(0, 1)).reshape(num_anchor_points, self.num_transformations, -1)
        original_output = self.model.deterministic_forward(x).flatten(start_dim=1).unsqueeze(1)
        response = transformed_output - original_output

        if was_training:
            self.model.train()

        return response

    def gram_matrix(self, anchor_indices=None):
        response_vector = self.model_response(anchor_indices)
        output_dimension = response_vector.shape[-1]

        g_matrix = (response_vector @ response_vector.transpose(-1, -2)) / output_dimension
        g_matrix = g_matrix / self.input_scale_matrix

        trace = torch.diagonal(g_matrix, dim1=-2, dim2=-1).sum(dim=-1)
        g_matrix = g_matrix / trace.clamp_min(1e-12).view(-1, 1, 1)

        return g_matrix

    def forward(self):
        anchor_indices = self.sample_anchor_indices()

        current_gram_matrices = self.gram_matrix(anchor_indices)
        source_gram_matrices = self.source_gram_matrices[anchor_indices]

        regularizer_loss = torch.square(current_gram_matrices - source_gram_matrices).sum(dim=(-2, -1)).mean()

        return regularizer_loss