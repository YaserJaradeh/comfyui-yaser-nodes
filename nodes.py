from .types import any
from .upscale_utils import upscale_with_model
import nodes
import inspect
import logging

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


class GeneralSwitch:
    """
    Switch (Any) - Select from inputs based on an index. Inputs are added dynamically.
    Based on ImpactSwitch from ComfyUI-Impact-Pack.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Check if execution model version is supported (for newer ComfyUI versions)
        dyn_inputs = {"input1": (any, {"lazy": True, "tooltip": "Any input. When connected, one more input slot is added."})}
        
        # For newer versions of ComfyUI, use AllContainer to bypass validation during input info gathering
        try:
            import comfy_execution  # noqa: F401
            execution_model_supported = True
        except Exception:
            execution_model_supported = False
            
        if execution_model_supported:
            stack = inspect.stack()
            if stack[2].function == 'get_input_info':
                # bypass validation
                class AllContainer:
                    def __contains__(self, item):
                        return True

                    def __getitem__(self, key):
                        return any, {"lazy": True}

                dyn_inputs = AllContainer()

        inputs = {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 999999, "step": 1, "tooltip": "The input number you want to output among the inputs"}),
                "sel_mode": ("BOOLEAN", {"default": False, "label_on": "select_on_prompt", "label_off": "select_on_execution", "forceInput": False,
                             "tooltip": "In the case of 'select_on_execution', the selection is dynamically determined at the time of workflow execution. 'select_on_prompt' is an option that exists for older versions of ComfyUI, and it makes the decision before the workflow execution."}),
            },
            "optional": dyn_inputs,
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}
        }

        return inputs

    RETURN_TYPES = (any, "STRING", "INT")
    RETURN_NAMES = ("selected_value", "selected_label", "selected_index")
    OUTPUT_TOOLTIPS = ("Output is generated only from the input chosen by the 'select' value.", "Slot label of the selected input slot", "Outputs the select value as is")
    FUNCTION = "doit"
    CATEGORY = "logic/switch"

    def check_lazy_status(self, *args, **kwargs):
        selected_index = int(kwargs['select'])
        input_name = f"input{selected_index}"

        logging.info(f"SELECTED: {input_name}")

        if input_name in kwargs:
            return [input_name]
        else:
            return []

    @staticmethod
    def doit(*args, **kwargs):
        selected_index = int(kwargs['select'])
        input_name = f"input{selected_index}"

        selected_label = input_name
        node_id = kwargs['unique_id']

        if 'extra_pnginfo' in kwargs and kwargs['extra_pnginfo'] is not None:
            nodelist = kwargs['extra_pnginfo']['workflow']['nodes']
            for node in nodelist:
                if str(node['id']) == node_id:
                    inputs = node['inputs']

                    for slot in inputs:
                        if slot['name'] == input_name and 'label' in slot:
                            selected_label = slot['label']

                    break
        else:
            logging.info("[Yaser-nodes] The switch node does not guarantee proper functioning in API mode.")

        if input_name in kwargs:
            return kwargs[input_name], selected_label, selected_index
        else:
            logging.info("GeneralSwitch: invalid select index (ignored)")
            return None, "", selected_index
