from .upscale_utils import upscale_with_model


class IterativeUpscaleWithModelsNode:
    DESCRIPTION = "Iteratively upscale an image using multiple models."
    
    def __init__(self):
        pass
    
    @classmethod  
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ([1, 2, 4, 8],),
                "model1x": ("UPSCALE_MODEL",),
                "model2x": ("UPSCALE_MODEL",),
                "model4x": ("UPSCALE_MODEL",),
                "model8x": ("UPSCALE_MODEL",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    
    FUNCTION = "upscale_image"

    CATEGORY = "image/upscaling"

    def upscale_image(self, image, scale_factor, model1x, model2x, model4x, model8x):
        """
        Upscale the provided image using the specified models.
        """
        # Select the appropriate model based on scale_factor
        if scale_factor == 1:
            selected_model = model1x
        elif scale_factor == 2:
            selected_model = model2x
        elif scale_factor == 4:
            selected_model = model4x
        else:  # scale_factor == 8
            selected_model = model8x

        # Use the upscale utility function
        return upscale_with_model(image, selected_model)
