import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRA(nn.Module):
    def __init__(self, base_layer, rank, alpha):
        super().__init__()

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merged = False
        self.adapters_enabled = True

        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

    def initialize_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        if self.merged or not self.adapters_enabled:
            return self.base_layer(x)

        return self.base_layer(x) + self.scaling * self.adapter_forward(x)

    def adapter_forward(self, x):
        raise NotImplementedError

    def delta_weight(self):
        raise NotImplementedError
    
    def enable_adapter(self):
        self.adapters_enabled = True

    def disable_adapter(self):
        self.adapters_enabled = False

    def merge(self):
        if self.merged:
            raise RuntimeError("LoRA weights are already merged, cannot merge again.")
        
        with torch.no_grad():
            self.base_layer.weight.add_(self.delta_weight())

        self.merged = True

        return self

    def unmerge(self):
        if not self.merged:
            raise RuntimeError("LoRA weights are not merged, cannot unmerge.")
        
        with torch.no_grad():
            self.base_layer.weight.sub_(self.delta_weight())

        self.merged = False

        return self


class LoRALinear(LoRA):
    def __init__(self, base_layer, rank, alpha):
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear requires an nn.Linear base layer")

        super().__init__(base_layer, rank, alpha)

        self.lora_A = nn.Parameter(base_layer.weight.new_empty(rank, base_layer.in_features))
        self.lora_B = nn.Parameter(base_layer.weight.new_empty(base_layer.out_features, rank))
        self.initialize_parameters()

    def adapter_forward(self, x):
        return F.linear(F.linear(x, self.lora_A), self.lora_B)

    def delta_weight(self):
        return self.scaling * (self.lora_B @ self.lora_A)


class LoRAConv2d(LoRA):
    def __init__(self, base_layer, rank, alpha):
        if not isinstance(base_layer, nn.Conv2d):
            raise TypeError("LoRAConv2d requires an nn.Conv2d base layer")

        if base_layer.groups != 1:
            raise NotImplementedError("LoRAConv2d currently supports only groups=1")

        if base_layer.padding_mode != "zeros":
            raise NotImplementedError("LoRAConv2d currently supports only padding_mode='zeros'")

        super().__init__(base_layer, rank, alpha)

        in_channels = base_layer.in_channels
        out_channels = base_layer.out_channels
        kH, kW = base_layer.kernel_size

        self.lora_A = nn.Parameter(base_layer.weight.new_empty(rank, in_channels, kH, kW))
        self.lora_B = nn.Parameter(base_layer.weight.new_empty(out_channels, rank, 1, 1))
        self.initialize_parameters()

    def adapter_forward(self, x):
        lora_A_out = F.conv2d(x, self.lora_A, bias=None, stride=self.base_layer.stride, padding=self.base_layer.padding, dilation=self.base_layer.dilation, groups=1)
        return F.conv2d(lora_A_out, self.lora_B, bias=None, stride=1, padding=0)

    def delta_weight(self):
        return self.scaling * torch.einsum("or,rihw->oihw", self.lora_B[:, :, 0, 0], self.lora_A)


class LoRAConvTranspose2d(LoRA):
    def __init__(self, base_layer, rank, alpha):
        if not isinstance(base_layer, nn.ConvTranspose2d):
            raise TypeError("LoRAConvTranspose2d requires an nn.ConvTranspose2d base layer")

        if base_layer.groups != 1:
            raise NotImplementedError("LoRAConvTranspose2d currently supports only groups=1")

        super().__init__(base_layer, rank, alpha)

        in_channels = base_layer.in_channels
        out_channels = base_layer.out_channels
        kH, kW = base_layer.kernel_size

        self.lora_A = nn.Parameter(base_layer.weight.new_empty(in_channels, rank, kH, kW))
        self.lora_B = nn.Parameter(base_layer.weight.new_empty(out_channels, rank, 1, 1))
        self.initialize_parameters()

    def adapter_forward(self, x):
        lora_A_out = F.conv_transpose2d(x, self.lora_A, bias=None, stride=self.base_layer.stride, padding=self.base_layer.padding, output_padding=self.base_layer.output_padding, groups=1, dilation=self.base_layer.dilation)
        return F.conv2d(lora_A_out, self.lora_B, bias=None, stride=1, padding=0)

    def delta_weight(self):
        return self.scaling * torch.einsum("or,irhw->iohw", self.lora_B[:, :, 0, 0], self.lora_A)