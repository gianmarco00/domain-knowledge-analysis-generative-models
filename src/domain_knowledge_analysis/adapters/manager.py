import torch
from .lora import LoRALinear, LoRAConv2d, LoRAConvTranspose2d
from domain_knowledge_analysis.utils import select_layers

class LoRAManager:
    def __init__(self, model, config):
        self.model = model
        self.rank = config["lora"]["rank"]
        self.alpha = config["lora"]["alpha"]
        self.config = config
        self.layers_to_inject = select_layers(self.model, self.config["lora"]["layers_to_inject"])

        self.injected = False
        self.adapters_enabled = True
        self.lora_layers = {}

    def inject_adapters(self):
        if self.injected:
            raise RuntimeError("LoRA has already been injected into the model.")

        for name, layer in self.layers_to_inject.items():
            lora_layer = self.create_lora_layer(layer)
            self.model.set_submodule(name, lora_layer, strict=True)
            self.lora_layers[name] = lora_layer

        self.injected = True

    def trainable_parameters(self):
        if not self.injected:
            raise RuntimeError("LoRA has not been injected into the model yet.")

        trainable_params = []
        for layer in self.lora_layers.values():
            trainable_params.extend(layer.lora_A)
            trainable_params.extend(layer.lora_B)

        return trainable_params

    def merge(self):
        if not self.injected:
            raise RuntimeError("LoRA has not been injected into the model yet.")

        for layer in self.lora_layers.values():
            layer.merge()
        
        return self.model
    
    def unmerge(self):
        if not self.injected:
            raise RuntimeError("LoRA has not been injected into the model yet.")

        for layer in self.lora_layers.values():
            layer.unmerge()
        
        return self.model
    
    def enable_adapters(self):
        for layer in self.lora_layers.values():
            layer.enable_adapter()
        self.adapters_enabled = True

    def disable_adapters(self):
        for layer in self.lora_layers.values():
            layer.disable_adapter()
        self.adapters_enabled = False

            

    def create_lora_layer(self, layer):

        if isinstance(layer, torch.nn.Linear):
            return LoRALinear(layer, self.rank, self.alpha)
        elif isinstance(layer, torch.nn.Conv2d):
            return LoRAConv2d(layer, self.rank, self.alpha)
        elif isinstance(layer, torch.nn.ConvTranspose2d):
            return LoRAConvTranspose2d(layer, self.rank, self.alpha)
        else:
            raise TypeError(f"Unsupported layer type: {type(layer)}")