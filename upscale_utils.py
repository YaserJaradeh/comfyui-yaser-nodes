"""
Upscale utilities for Yaser-nodes
Contains the core upscaling logic copied from ComfyUI's ImageUpscaleWithModel
Based on https://raw.githubusercontent.com/comfyanonymous/ComfyUI/d2aaef029cfb60611f2c9aad1b5dfb7070f9c162/comfy_extras/nodes_upscale_model.py
"""

import comfy.model_management as model_management
import comfy.utils
import torch


def upscale_with_model(image, upscale_model):
    """
    Upscale an image using the provided upscale model.
    
    Args:
        image: Input image tensor in NHWC format
        upscale_model: The upscale model to use
    
    Returns:
        Tuple containing the upscaled image tensor
    """
    # Core upscale logic copied from ComfyUI's ImageUpscaleWithModel
    device = model_management.get_torch_device()

    # Calculate memory requirements
    memory_required = model_management.module_size(upscale_model.model)
    memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
    memory_required += image.nelement() * image.element_size()
    model_management.free_memory(memory_required, device)

    # Move model to device
    upscale_model.to(device)
    
    # Convert image from NHWC to NCHW format (ComfyUI uses movedim(-1,-3))
    in_img = image.movedim(-1, -3).to(device)

    # Upscale parameters
    tile = 512
    overlap = 32

    # Handle out-of-memory by reducing tile size
    oom = True
    while oom:
        try:
            steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
            )
            pbar = comfy.utils.ProgressBar(steps)
            
            # Perform the actual upscaling
            s = comfy.utils.tiled_scale(
                in_img, 
                lambda a: upscale_model(a),
                tile_x=tile, 
                tile_y=tile, 
                overlap=overlap, 
                upscale_amount=upscale_model.scale, 
                pbar=pbar
            )
            oom = False
        except model_management.OOM_EXCEPTION as e:
            tile //= 2
            if tile < 128:
                raise e

    # Move model back to CPU to free GPU memory
    upscale_model.to("cpu")
    
    # Convert back to NHWC format and clamp values
    s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
    
    return (s,)
