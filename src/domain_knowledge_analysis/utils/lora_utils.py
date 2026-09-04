import torch
import torch.nn as nn

LORA_LAYER_TYPES = {
    "fc": nn.Linear,
    "conv": nn.Conv2d,
    "convtranspose": nn.ConvTranspose2d,
}

def select_layers(model, layers_to_select):

    selected_layers = {}

    for section_name, layers in layers_to_select.items():
        
        section = model.get_submodule(section_name)

        layers_type = tuple(LORA_LAYER_TYPES[layer] for layer in layers)

        for relative_name, module in section.named_modules():
            if relative_name == "":
                continue

            if isinstance(module, layers_type):
                full_name = f"{section_name}.{relative_name}"
                selected_layers[full_name] = module

    return selected_layers
