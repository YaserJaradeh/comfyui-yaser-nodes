from .nodes import IterativeUpscaleWithModelsNode, GeneralSwitch, ImageColorMatch, LoraLoaderStack, ImageSizeBySide, Float, TiledDiffusionNode, SpotDiffusionParams_Yaser, TiledVAEEncode, TiledVAEDecode, UtilRepeatImages

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
}

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
}
