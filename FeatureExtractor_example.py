class FeatureExtractor:
    def __init__(self, model, layers: str):
        self.model = model.model
        self.layers = layers
        self.features = {layer: torch.empty(0) for layer in layers}
        self.hooks = []

    def get_hooks(self):
        # You can modify this part, it's dependent on your model.
        for layer_id in self.layers:
            layer = self.model._modules[layer_id]
            self.hooks.append(layer.register_forward_hook(self.save_outputs_hook(layer_id)))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def save_outputs_hook(self, layer_id: str):
        def fn(_, __, output):
            self.features[layer_id] = output

        return fn
