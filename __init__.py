from .nodes import IterativeUpscaleWithModelsNode, GeneralSwitch, ImageColorMatch, LoraLoaderStack, ImageSizeBySide, Float, TiledDiffusionNode, SpotDiffusionParams, TiledVAEEncode, TiledVAEDecode, UtilRepeatImages, WanVideoNAG
from .nodes.controlnet import NODE_CLASS_MAPPINGS as CONTROLNET_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as CONTROLNET_NODE_DISPLAY_NAME_MAPPINGS

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "IterativeUpscaleWithModelsNode": IterativeUpscaleWithModelsNode,
    "GeneralSwitch": GeneralSwitch,
    "ImageColorMatch": ImageColorMatch,
    "LoraLoaderStack": LoraLoaderStack,
    "ImageSizeBySide": ImageSizeBySide,
    "Float": Float,
    "TiledDiffusion": TiledDiffusionNode,
    "SpotDiffusionParams": SpotDiffusionParams,
    "TiledVAEEncode": TiledVAEEncode,
    "TiledVAEDecode": TiledVAEDecode,
    "UtilRepeatImages": UtilRepeatImages,
    "WanVideoNAG": WanVideoNAG,
}

# Add controlnet nodes
NODE_CLASS_MAPPINGS.update(CONTROLNET_NODE_CLASS_MAPPINGS)

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "IterativeUpscaleWithModelsNode": "🔮 Iterative Upscale with Models Node (Yaser)",
    "GeneralSwitch": "🔀 Switch (Any) - Yaser",
    "ImageColorMatch": "🎨 Image Color Match - Yaser",
    "LoraLoaderStack": "📚 LoRA Loader Stack - Yaser",
    "ImageSizeBySide": "📏 ImageSize (Side) - Yaser",
    "Float": "🔢 Float - Yaser",
    "TiledDiffusion": "🎨 Tiled Diffusion - Yaser",
    "SpotDiffusionParams": "🎯 SpotDiffusion Parameters - Yaser",
    "TiledVAEEncode": "📦 Tiled VAE Encode - Yaser",
    "TiledVAEDecode": "📦 Tiled VAE Decode - Yaser",
    "UtilRepeatImages": "🔁 Repeat Images - Yaser",
    "WanVideoNAG": "📹 Wan Video NAG - Yaser",
}

# Add controlnet display names
NODE_DISPLAY_NAME_MAPPINGS.update(CONTROLNET_NODE_DISPLAY_NAME_MAPPINGS)
