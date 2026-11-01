import math

import torch

from domain_knowledge_analysis.utils.utils import INPUT_SCALE_FLOOR, RMS_ABSOLUTE_TOLERANCE, RMS_RELATIVE_TOLERANCE


class TFRegularizer(torch.nn.Module):
    VALID_MODES = {"output_gram", "joint_gram", "encoder_response", "encoder_gram", "source_loss"}
    JOINT_ENERGY_FLOOR = 1e-12

    def __init__(self, model, lora_manager, transformations, anchor_points, anchor_batch_size=10, calibration_batch_size=100, target_input_rms=None, target_input_scales=None, mode="output_gram", response_weight=1.0, decoder_weight=0.0, decoder_seed=0, loss=None, trace_normalization=True, fingerprint_map="reconstruction"):
        super().__init__()

        if mode not in self.VALID_MODES:
            raise ValueError(f"Regularizer mode must be one of {sorted(self.VALID_MODES)}.")
        if not math.isfinite(response_weight) or response_weight < 0:
            raise ValueError("Regularizer response_weight must be finite and nonnegative.")
        if not math.isfinite(decoder_weight) or decoder_weight < 0:
            raise ValueError("Regularizer decoder_weight must be finite and nonnegative.")
        if not isinstance(trace_normalization, bool):
            raise ValueError("Regularizer trace_normalization must be a boolean.")
        if fingerprint_map not in {"reconstruction", "fixed_decoder"}:
            raise ValueError("Regularizer fingerprint_map must be reconstruction or fixed_decoder.")
        if fingerprint_map != "reconstruction" and mode not in {"output_gram", "joint_gram"}:
            raise ValueError("fixed_decoder fingerprint_map requires output_gram or joint_gram mode.")
        if not isinstance(anchor_batch_size, int) or isinstance(anchor_batch_size, bool) or anchor_batch_size < 1:
            raise ValueError("Regularizer anchor_batch_size must be a positive integer.")

        self.model = model
        self.lora_manager = lora_manager
        self.mode = mode
        self.fingerprint_map = fingerprint_map
        self.trace_normalization = trace_normalization
        self.response_weight = float(response_weight)
        self.decoder_weight = float(decoder_weight)
        self.decoder_seed = int(decoder_seed)
        self.x = anchor_points
        self.num_anchor_points = len(self.x)
        self.anchor_batch_size = anchor_batch_size
        self.calibration_batch_size = calibration_batch_size
        if self.mode == "joint_gram":
            self.anchor_batch_size = min(anchor_batch_size, self.num_anchor_points)
            if self.anchor_batch_size < 2:
                raise ValueError("joint_gram requires at least two anchors per batch.")

        if self.mode == "source_loss":
            if loss is None:
                raise ValueError("source_loss mode requires the training loss.")

            self.loss = loss
            self.shuffle_anchor_indices()
            return

        self.transformations_functions, _ = transformations
        self.num_transformations = len(self.transformations_functions)
        self.target_input_rms = float(target_input_rms) if target_input_rms is not None else None
        self.target_input_scales = torch.as_tensor(target_input_scales, dtype=self.x.dtype, device=self.x.device) if target_input_scales is not None else None
        self.transformed_x = self.apply_transform(self.x)

        self.shuffle_anchor_indices()
        self.calibrate()
        if self.mode == "joint_gram":
            # Check once that even the lowest-energy possible batch is valid.
            energies = self.source_joint_responses.square().sum(dim=(1, 2))
            minimum_batch_energy = energies.topk(self.anchor_batch_size, largest=False).values.sum()
            if not torch.isfinite(energies).all() or minimum_batch_energy <= self.JOINT_ENERGY_FLOOR:
                raise ValueError("joint_gram requires finite, nonzero source response energy for every anchor batch.")

    def calibrate(self):
        self.calibrate_input_scale()

        was_training = self.model.training
        adapters_were_enabled = self.lora_manager.adapters_enabled
        self.model.eval()
        self.lora_manager.disable_adapters()

        try:
            with torch.no_grad():
                if self.fingerprint_map == "fixed_decoder":
                    self.calibrate_fixed_decoder_probes()
                if self.mode == "output_gram":
                    self.source_gram_matrices = self.gram_matrix_batched(self.calibration_batch_size)
                elif self.mode == "joint_gram":
                    responses = []
                    for start in range(0, self.num_anchor_points, self.calibration_batch_size):
                        indices = torch.arange(start, min(start + self.calibration_batch_size, self.num_anchor_points), device=self.x.device)
                        responses.append(self.model_response(indices) / self.input_scales.view(1, -1, 1))
                    self.source_joint_responses = torch.cat(responses)
                else:
                    source_encoder_outputs = []
                    source_transformation_geometries = []
                    source_latents = []
                    source_decoder_outputs = []

                    if self.decoder_weight > 0:
                        latent_dim = self.model.encoder_params["latent_dim"]
                        generator = torch.Generator(device="cpu")
                        generator.manual_seed(self.decoder_seed)
                        posterior_noise = torch.randn(self.num_anchor_points, latent_dim, generator=generator, dtype=self.x.dtype).to(self.x.device)
                        prior_latents = torch.randn(self.num_anchor_points, latent_dim, generator=generator, dtype=self.x.dtype).to(self.x.device)

                    for start in range(0, self.num_anchor_points, self.calibration_batch_size):
                        end = min(start + self.calibration_batch_size, self.num_anchor_points)
                        anchor_indices = torch.arange(start, end, device=self.x.device)
                        encoder_output, response = self.model_response(anchor_indices, return_original=True)
                        source_encoder_outputs.append(encoder_output)
                        source_transformation_geometries.append(self.transformation_geometry(response))

                        if self.decoder_weight > 0:
                            mean, std = encoder_output.chunk(2, dim=-1)
                            latents = torch.stack([mean, mean + std * posterior_noise[start:end], prior_latents[start:end]], dim=1)
                            source_latents.append(latents)
                            source_decoder_outputs.append(self.decoder_output(latents.flatten(0, 1)).reshape(end - start, 3, -1))

                    self.source_encoder_outputs = torch.cat(source_encoder_outputs)
                    self.source_transformation_geometries = torch.cat(source_transformation_geometries)

                    if self.decoder_weight > 0:
                        self.source_latents = torch.cat(source_latents)
                        self.source_decoder_outputs = torch.cat(source_decoder_outputs)

        finally:
            if adapters_were_enabled:
                self.lora_manager.enable_adapters()
            self.model.train(was_training)

    def calibrate_fixed_decoder_probes(self):
        probes = []
        for start in range(0, self.num_anchor_points, self.calibration_batch_size):
            x = self.x[start:start + self.calibration_batch_size]
            transformed = self.transformed_x[start:start + self.calibration_batch_size]
            images = torch.cat((x.unsqueeze(1), transformed), dim=1)
            means, _ = self.model.encoder(images.flatten(0, 1))
            probes.append(means.reshape(len(x), self.num_transformations + 1, -1))
        self.fixed_decoder_probes = torch.cat(probes)

    def calibrate_input_scale(self):
        # This accumulates the numerator and denominator of input_delta.square().mean(dim=(0, 2)).
        input_square_sum = torch.zeros(self.num_transformations, dtype=self.x.dtype, device=self.x.device)
        num_input_values = 0
        per_anchor_input_rms = []

        for start in range(0, self.num_anchor_points, self.calibration_batch_size):
            x = self.x[start:start + self.calibration_batch_size]
            transformed_x = self.transformed_x[start:start + self.calibration_batch_size]
            input_delta = (transformed_x - x.unsqueeze(1)).flatten(start_dim=2)

            input_square_sum += input_delta.square().sum(dim=(0, 2))
            num_input_values += len(x) * input_delta.shape[-1]
            per_anchor_input_rms.append(input_delta.square().mean(dim=2).sqrt())

        self.input_scales = (input_square_sum / num_input_values).sqrt()
        self.per_anchor_input_rms = torch.cat(per_anchor_input_rms)

        if not torch.isfinite(self.input_scales).all() or (self.input_scales <= INPUT_SCALE_FLOOR).any():
            raise ValueError("Every regularizer transformation must produce a finite input RMS greater than 1e-6.")

        if self.target_input_rms is not None:
            self.target_input_scales = torch.full_like(self.input_scales, float(self.target_input_rms))

        if self.target_input_scales is not None:
            if self.target_input_scales.shape != self.input_scales.shape or not torch.allclose(self.input_scales, self.target_input_scales, rtol=RMS_RELATIVE_TOLERANCE, atol=RMS_ABSOLUTE_TOLERANCE):
                raise ValueError(f"Measured regularizer input scales {self.input_scales.tolist()} do not match calibrated targets {self.target_input_scales.tolist()}.")

        self.input_scale_matrix = torch.outer(self.input_scales, self.input_scales)

    def shuffle_anchor_indices(self):
        self.anchor_indices = torch.randperm(self.num_anchor_points, device=self.x.device)
        self.anchor_position = 0

    def sample_anchor_indices(self):
        if self.mode == "joint_gram" and self.anchor_position + self.anchor_batch_size > self.num_anchor_points:
            # Discard the short remainder to keep the joint loss batch size fixed.
            self.shuffle_anchor_indices()
        start = self.anchor_position
        end = min(start + self.anchor_batch_size, self.num_anchor_points)

        anchor_indices = self.anchor_indices[start:end]
        self.anchor_position = end

        if self.anchor_position == self.num_anchor_points:
            self.shuffle_anchor_indices()

        return anchor_indices

    def apply_transform(self, x):
        with torch.no_grad():
            transformed_x = torch.stack([t(x) for t in self.transformations_functions], dim=1)

        return transformed_x

    def model_representation(self, x):
        if self.mode in {"output_gram", "joint_gram"}:
            return self.model.deterministic_forward(x)

        mean, log_variance = self.model.encoder(x)
        return torch.cat([mean, torch.exp(0.5 * log_variance)], dim=-1)

    def decoder_output(self, latents):
        logits = self.model.decoder(latents)
        return self.model.decoder_distribution(logits, self.model.decoder_distribution_name).flatten(start_dim=1)

    def model_response(self, anchor_indices=None, return_original=False):
        if self.fingerprint_map == "fixed_decoder":
            probes = self.fixed_decoder_probes if anchor_indices is None else self.fixed_decoder_probes[anchor_indices]
            was_training = self.model.training
            self.model.eval()
            try:
                outputs = self.decoder_output(probes.flatten(0, 1)).reshape(len(probes), self.num_transformations + 1, -1)
            finally:
                self.model.train(was_training)
            original = outputs[:, :1]
            response = outputs[:, 1:] - original
            return (original.squeeze(1), response) if return_original else response

        x = self.x if anchor_indices is None else self.x[anchor_indices]
        transformed_x = self.transformed_x if anchor_indices is None else self.transformed_x[anchor_indices]

        num_anchor_points = len(x)

        was_training = self.model.training
        self.model.eval()

        transformed_output = self.model_representation(transformed_x.flatten(0, 1)).reshape(num_anchor_points, self.num_transformations, -1)
        original_output = self.model_representation(x).flatten(start_dim=1).unsqueeze(1)
        response = transformed_output - original_output

        self.model.train(was_training)

        if return_original:
            return original_output.squeeze(1), response

        return response

    @classmethod
    def normalized_joint_gram(cls, scaled_responses):
        # Rows are (anchor, transformation); columns are output coordinates.
        rows = scaled_responses.flatten(0, 1)
        gram = rows @ rows.transpose(0, 1)
        energy = gram.diagonal().sum()
        return gram / energy.clamp_min(cls.JOINT_ENERGY_FLOOR)

    def gram_matrix(self, anchor_indices=None, response_vector=None):
        response_vector = self.model_response(anchor_indices) if response_vector is None else response_vector
        output_dimension = response_vector.shape[-1]

        g_matrix = (response_vector @ response_vector.transpose(-1, -2)) / output_dimension
        g_matrix = g_matrix / self.input_scale_matrix

        if self.mode == "output_gram" and self.trace_normalization:
            trace = torch.diagonal(g_matrix, dim1=-2, dim2=-1).sum(dim=-1)
            g_matrix = g_matrix / trace.clamp_min(1e-12).view(-1, 1, 1)

        return g_matrix

    def transformation_geometry(self, response_vector):
        if self.mode == "encoder_response":
            return response_vector / self.input_scales.view(1, -1, 1)

        return self.gram_matrix(response_vector=response_vector)

    def gram_matrix_batched(self, batch_size=None):
        batch_size = batch_size or self.calibration_batch_size
        gram_matrices = []

        for start in range(0, self.num_anchor_points, batch_size):
            anchor_indices = torch.arange(start, min(start + batch_size, self.num_anchor_points), device=self.x.device)
            gram_matrices.append(self.gram_matrix(anchor_indices))

        return torch.cat(gram_matrices)

    def forward(self):
        anchor_indices = self.sample_anchor_indices()

        if self.mode == "source_loss":
            source_x = self.x[anchor_indices]
            return self.loss(source_x, *self.model(source_x))

        if self.mode == "output_gram":
            current_gram_matrices = self.gram_matrix(anchor_indices)
            source_gram_matrices = self.source_gram_matrices[anchor_indices]
            return torch.square(current_gram_matrices - source_gram_matrices).mean()

        if self.mode == "joint_gram":
            current = self.model_response(anchor_indices) / self.input_scales.view(1, -1, 1)
            source = self.source_joint_responses[anchor_indices]
            current_gram = self.normalized_joint_gram(current)
            source_gram = self.normalized_joint_gram(source)
            return torch.square(current_gram - source_gram).sum()

        was_training = self.model.training
        self.model.eval()

        encoder_output, response = self.model_response(anchor_indices, return_original=True)
        transformation_geometry = self.transformation_geometry(response)
        encoder_loss = torch.square(encoder_output - self.source_encoder_outputs[anchor_indices]).mean()
        response_loss = torch.square(transformation_geometry - self.source_transformation_geometries[anchor_indices]).mean()
        regularizer_loss = encoder_loss + self.response_weight * response_loss

        if self.decoder_weight > 0:
            latents = self.source_latents[anchor_indices]
            decoder_output = self.decoder_output(latents.flatten(0, 1)).reshape(len(anchor_indices), 3, -1)
            decoder_loss = torch.square(decoder_output - self.source_decoder_outputs[anchor_indices]).mean()
            regularizer_loss = regularizer_loss + self.decoder_weight * decoder_loss

        self.model.train(was_training)
        return regularizer_loss
