"""
Image utility nodes for Yaser-nodes
Contains image processing and analysis nodes
"""


class ImageSizeBySide:
    """
    ImageSize (Side) - Get the longest or shortest side dimension of an image.
    Based on imageSizeBySide from ComfyUI-Easy-Use.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "side": (["Longest", "Shortest"],)
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("resolution",)
    OUTPUT_NODE = True
    FUNCTION = "image_side"
    CATEGORY = "Yaser/Image"
    DESCRIPTION = "Get the longest or shortest side dimension of an image."

    def image_side(self, image, side):
        """Get the specified side dimension of the image"""
        _, raw_H, raw_W, _ = image.shape

        width = raw_W
        height = raw_H
        
        if width is not None and height is not None:
            if side == "Longest":
                result = (width,) if width > height else (height,)
            elif side == 'Shortest':
                result = (width,) if width < height else (height,)
        else:
            result = (0,)
            
        return {"ui": {"text": str(result[0])}, "result": result}
