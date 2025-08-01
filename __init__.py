from .nodes import IterativeUpscaleWithModelsNode, GeneralSwitch, ImageColorMatch

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "IterativeUpscaleWithModelsNode": IterativeUpscaleWithModelsNode,
    "GeneralSwitch": GeneralSwitch,
    "ImageColorMatch": ImageColorMatch
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "IterativeUpscaleWithModelsNode": "🔮 Iterative Upscale with Models Node (Yaser)",
    "GeneralSwitch": "🔀 Switch (Any) - Yaser",
    "ImageColorMatch": "🎨 Image Color Match - Yaser"
}