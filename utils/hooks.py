
def register_clip_hooks(model, layer_names):
    """
    Registers forward hooks for specified layer names in a CLIP model.
    Returns:
        activations: dict of {layer_name: output tensor}
        handles: list of hook handles (for later removal)
    """
    activations = {}
    handles = []
    all_modules = dict(model.named_modules())
    for name in layer_names:
        def get_hook(n):
            return lambda module, inp, out: activations.setdefault(n, out.detach())
        handle = all_modules[name].register_forward_hook(get_hook(name))
        handles.append(handle)
    return activations, handles