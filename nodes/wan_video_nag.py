import comfy.ldm.modules.attention
import torch
import comfy.model_management as mm
import types


def normalized_attention_guidance(self, query, context_positive, context_negative, transformer_options={}):
    k_positive = self.norm_k(self.k(context_positive))
    v_positive = self.v(context_positive)
    k_negative = self.norm_k(self.k(context_negative))
    v_negative = self.v(context_negative)

    try:
        x_positive = comfy.ldm.modules.attention.optimized_attention(query, k_positive, v_positive, heads=self.num_heads, transformer_options=transformer_options).flatten(2)
        x_negative = comfy.ldm.modules.attention.optimized_attention(query, k_negative, v_negative, heads=self.num_heads, transformer_options=transformer_options).flatten(2)
    except: #backwards compatibility for now
        x_positive = comfy.ldm.modules.attention.optimized_attention(query, k_positive, v_positive, heads=self.num_heads).flatten(2)
        x_negative = comfy.ldm.modules.attention.optimized_attention(query, k_negative, v_negative, heads=self.num_heads).flatten(2)

    nag_guidance = x_positive * self.nag_scale - x_negative * (self.nag_scale - 1)

    norm_positive = torch.norm(x_positive, p=1, dim=-1, keepdim=True).expand_as(x_positive)
    norm_guidance = torch.norm(nag_guidance, p=1, dim=-1, keepdim=True).expand_as(nag_guidance)
    
    scale = torch.nan_to_num(norm_guidance / norm_positive, nan=10.0)

    mask = scale > self.nag_tau
    adjustment = (norm_positive * self.nag_tau) / (norm_guidance + 1e-7)
    nag_guidance = torch.where(mask, nag_guidance * adjustment, nag_guidance)

    x = nag_guidance * self.nag_alpha + x_positive * (1 - self.nag_alpha)
    del nag_guidance

    return x


def wan_crossattn_forward_nag(self, x, context, transformer_options={}, **kwargs):
    r"""
    Args:
        x(Tensor): Shape [B, L1, C]
        context(Tensor): Shape [B, L2, C]
    """
    # Determine batch splitting and context handling
    if self.input_type == "default":
        # Single or [pos, neg] pair
        if context.shape[0] == 1:
            x_pos, context_pos = x, context
            x_neg, context_neg = None, None
        else:
            x_pos, x_neg = torch.chunk(x, 2, dim=0)
            context_pos, context_neg = torch.chunk(context, 2, dim=0)
    elif self.input_type == "batch":
        # Standard batch, no CFG
        x_pos, context_pos = x, context
        x_neg, context_neg = None, None

    # Positive branch
    q_pos = self.norm_q(self.q(x_pos))
    nag_context = self.nag_context
    if self.input_type == "batch":
        nag_context = nag_context.repeat(x_pos.shape[0], 1, 1)
    try:
        x_pos_out = normalized_attention_guidance(self, q_pos, context_pos, nag_context, transformer_options=transformer_options)
    except: #backwards compatibility for now
        x_pos_out = normalized_attention_guidance(self, q_pos, context_pos, nag_context)

    # Negative branch
    if x_neg is not None and context_neg is not None:
        q_neg = self.norm_q(self.q(x_neg))
        k_neg = self.norm_k(self.k(context_neg))
        v_neg = self.v(context_neg)
        try:
            x_neg_out = comfy.ldm.modules.attention.optimized_attention(q_neg, k_neg, v_neg, heads=self.num_heads, transformer_options=transformer_options)
        except: #backwards compatibility for now
            x_neg_out = comfy.ldm.modules.attention.optimized_attention(q_neg, k_neg, v_neg, heads=self.num_heads)
        x = torch.cat([x_pos_out, x_neg_out], dim=0)
    else:
        x = x_pos_out

    return self.o(x)


def wan_i2v_crossattn_forward_nag(self, x, context, context_img_len, transformer_options={}, **kwargs):
    r"""
    Args:
        x(Tensor): Shape [B, L1, C]
        context(Tensor): Shape [B, L2, C]
    """
    context_img = context[:, :context_img_len]
    context = context[:, context_img_len:]

    q_img = self.norm_q(self.q(x))    
    k_img = self.norm_k_img(self.k_img(context_img))
    v_img = self.v_img(context_img)
    try:
        img_x = comfy.ldm.modules.attention.optimized_attention(q_img, k_img, v_img, heads=self.num_heads, transformer_options=transformer_options)
    except: #backwards compatibility for now
        img_x = comfy.ldm.modules.attention.optimized_attention(q_img, k_img, v_img, heads=self.num_heads)

    if context.shape[0] == 2:
        x, x_real_negative = torch.chunk(x, 2, dim=0)
        context_positive, context_negative = torch.chunk(context, 2, dim=0)
    else:
        context_positive = context
        context_negative = None
    
    q = self.norm_q(self.q(x))

    x = normalized_attention_guidance(self, q, context_positive, self.nag_context, transformer_options=transformer_options)

    if context_negative is not None:
        q_real_negative = self.norm_q(self.q(x_real_negative))
        k_real_negative = self.norm_k(self.k(context_negative))
        v_real_negative = self.v(context_negative)
        try:
            x_real_negative = comfy.ldm.modules.attention.optimized_attention(q_real_negative, k_real_negative, v_real_negative, heads=self.num_heads, transformer_options=transformer_options)
        except: #backwards compatibility for now
            x_real_negative = comfy.ldm.modules.attention.optimized_attention(q_real_negative, k_real_negative, v_real_negative, heads=self.num_heads)
        x = torch.cat([x, x_real_negative], dim=0)

    # output
    x = x + img_x
    x = self.o(x)
    return x


class WanCrossAttentionPatch:
    def __init__(self, context, nag_scale, nag_alpha, nag_tau, i2v=False, input_type="default"):
        self.nag_context = context
        self.nag_scale = nag_scale
        self.nag_alpha = nag_alpha
        self.nag_tau = nag_tau
        self.i2v = i2v
        self.input_type = input_type
    def __get__(self, obj, objtype=None):
        # Create bound method with stored parameters
        def wrapped_attention(self_module, *args, **kwargs):
            self_module.nag_context = self.nag_context
            self_module.nag_scale = self.nag_scale
            self_module.nag_alpha = self.nag_alpha
            self_module.nag_tau = self.nag_tau
            self_module.input_type = self.input_type
            if self.i2v:
                return wan_i2v_crossattn_forward_nag(self_module, *args, **kwargs)
            else:
                return wan_crossattn_forward_nag(self_module, *args, **kwargs)
        return types.MethodType(wrapped_attention, obj)
    

class WanVideoNAG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "nag_scale": ("FLOAT", {"default": 11.0, "min": 0.0, "max": 100.0, "step": 0.001, "tooltip": "Strength of negative guidance effect"}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Mixing coefficient in that controls the balance between the normalized guided representation and the original positive representation."}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Clipping threshold that controls how much the guided attention can deviate from the positive attention."}),
           },
           "optional": {
                "input_type": (["default", "batch"], {"tooltip": "Type of the model input"}),
           },
                                                 
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = "https://github.com/ChenDarYen/Normalized-Attention-Guidance"
    EXPERIMENTAL = True

    def patch(self, model, conditioning, nag_scale, nag_alpha, nag_tau, input_type="default"):
        if nag_scale == 0:
            return (model,)
        
        device = mm.get_torch_device()
        dtype = mm.unet_dtype()

        model_clone = model.clone()

        diffusion_model = model_clone.get_model_object("diffusion_model")

        diffusion_model.text_embedding.to(device)
        context = diffusion_model.text_embedding(conditioning[0][0].to(device, dtype))

        type_str = str(type(model.model.model_config).__name__)
        i2v = True if "WAN21_I2V" in type_str else False
    
        for idx, block in enumerate(diffusion_model.blocks):
            patched_attn = WanCrossAttentionPatch(context, nag_scale, nag_alpha, nag_tau, i2v, input_type=input_type).__get__(block.cross_attn, block.__class__)
          
            model_clone.add_object_patch(f"diffusion_model.blocks.{idx}.cross_attn.forward", patched_attn)
            
        return (model_clone,)


# Node class mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "WanVideoNAG": WanVideoNAG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoNAG": "Wan Video NAG",
}
