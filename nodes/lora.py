from nodes import LoraLoader
import folder_paths


class LoraLoaderStack:
    """
    Lora Loader Stack - Load multiple LoRA models in sequence.
    Based on RgthreeLoraLoaderStack from rgthree-comfy.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),

                "lora_01": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_01": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "lora_02": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_02": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "lora_03": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_03": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "lora_04": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_04": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "lora_05": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_05": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    CATEGORY = "loaders"
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "load_lora_stack"
    DESCRIPTION = "Load multiple LoRA models in sequence with individual strength controls."

    def load_lora_stack(self, model, clip, lora_01, strength_01, lora_02, strength_02, 
                       lora_03, strength_03, lora_04, strength_04, lora_05, strength_05):
        """Load multiple LoRA models in sequence"""
        
        # Initialize LoraLoader
        lora_loader = LoraLoader()
        
        # Apply each LoRA if it's not "None" and strength is not 0
        if lora_01 != "None" and strength_01 != 0:
            model, clip = lora_loader.load_lora(model, clip, lora_01, strength_01, strength_01)
        
        if lora_02 != "None" and strength_02 != 0:
            model, clip = lora_loader.load_lora(model, clip, lora_02, strength_02, strength_02)
        
        if lora_03 != "None" and strength_03 != 0:
            model, clip = lora_loader.load_lora(model, clip, lora_03, strength_03, strength_03)
        
        if lora_04 != "None" and strength_04 != 0:
            model, clip = lora_loader.load_lora(model, clip, lora_04, strength_04, strength_04)
        
        if lora_05 != "None" and strength_05 != 0:
            model, clip = lora_loader.load_lora(model, clip, lora_05, strength_05, strength_05)

        return (model, clip)
