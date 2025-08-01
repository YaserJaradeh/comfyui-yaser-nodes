import numpy as np
import subprocess
import sys
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import ToTensor, ToPILImage
import importlib.metadata


def install_package(package, v=None):
    """Install a package using pip"""
    try:
        importlib.metadata.version(package)
        return False  # Already installed
    except importlib.metadata.PackageNotFoundError:
        pass
    
    package_command = package + '==' + v if v is not None else package
    print(f"Installing {package_command}...")
    result = subprocess.run([sys.executable, '-s', '-m', 'pip', 'install', package_command], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Package {package} installed successfully")
        return True
    else:
        print(f"Package {package} install failed: {result.stderr}")
        return False

def pil2tensor(image):
    """Convert PIL image to tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    """Convert tensor to PIL image"""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def calc_mean_std(feat: Tensor, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization"""
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat: Tensor, style_feat: Tensor):
    """Adaptive instance normalization for color matching"""
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def wavelet_blur(image: Tensor, radius: int):
    """Apply wavelet blur to the input tensor"""
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None]
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output

def wavelet_decomposition(image: Tensor, levels=5):
    """Apply wavelet decomposition to the input tensor"""
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq
    return high_freq, low_freq

def wavelet_reconstruction(content_feat: Tensor, style_feat: Tensor):
    """Apply wavelet reconstruction for color matching"""
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    return content_high_freq + style_low_freq

def adain_color_fix(target: Image, source: Image):
    """Apply adaptive instance normalization color fix"""
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)
    
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)
    
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))
    return result_image

def wavelet_color_fix(target: Image, source: Image):
    """Apply wavelet color fix"""
    source = source.resize(target.size, resample=Image.Resampling.LANCZOS)
    
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)
    
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)
    
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))
    return result_image

class ImageColorMatch:
    """
    Image Color Match - Transfer color characteristics from reference image to target image.
    Based on imageColorMatch from ComfyUI-Easy-Use.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "method": (['wavelet', 'adain', 'mkl', 'hm', 'reinhard', 'mvgd', 'hm-mvgd-hm', 'hm-mkl-hm'],),
            },
        }

    CATEGORY = "image/color"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "color_match"
    DESCRIPTION = "Transfer color characteristics from reference image to target image using various methods."

    def color_match(self, image_ref, image_target, method):
        """Match colors between reference and target images"""
        
        if method in ["wavelet", "adain"]:
            # Use built-in methods
            if method == 'wavelet':
                result_images = wavelet_color_fix(tensor2pil(image_target), tensor2pil(image_ref))
            else:  # adain
                result_images = adain_color_fix(tensor2pil(image_target), tensor2pil(image_ref))
            new_images = pil2tensor(result_images)
        else:
            # Use color-matcher library for other methods
            try:
                from color_matcher import ColorMatcher
            except ImportError:
                install_package("color-matcher")
                from color_matcher import ColorMatcher
            
            image_ref = image_ref.cpu()
            image_target = image_target.cpu()
            batch_size = image_target.size(0)
            out = []
            images_target = image_target.squeeze()
            images_ref = image_ref.squeeze()

            image_ref_np = images_ref.numpy()
            images_target_np = images_target.numpy()
            
            if image_ref.size(0) > 1 and image_ref.size(0) != batch_size:
                raise ValueError("ColorMatch: Use either single reference image or a matching batch of reference images.")
            
            cm = ColorMatcher()
            for i in range(batch_size):
                image_target_np = images_target_np if batch_size == 1 else images_target[i].numpy()
                image_ref_np_i = image_ref_np if image_ref.size(0) == 1 else images_ref[i].numpy()
                try:
                    image_result = cm.transfer(src=image_target_np, ref=image_ref_np_i, method=method)
                except BaseException as e:
                    print(f"Error occurred during transfer: {e}")
                    break
                out.append(torch.from_numpy(image_result))

            new_images = torch.stack(out, dim=0).to(torch.float32)

        return (new_images,)
