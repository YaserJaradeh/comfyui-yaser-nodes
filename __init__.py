from .nodes import ConditionalSelectionNode, IterativeUpscaleWithModelsNode

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "ConditionalSelectionNode": ConditionalSelectionNode,
    "IterativeUpscaleWithModelsNode": IterativeUpscaleWithModelsNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditionalSelectionNode": "🔮 Conditional Selection Node (Yaser)",
    "IterativeUpscaleWithModelsNode": "🔮 Iterative Upscale with Models Node (Yaser)",
}