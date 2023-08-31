from timm.models.xcit import Xcit, ClassAttentionBlock
from timm.models.cait import ClassAttn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, trunc_normal_, to_2tuple

from timm.models._registry import register_model, generate_default_cfgs, register_model_deprecations
from timm.models._builder import build_model_with_cfg

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from functools import partial
import math
import numpy as np
import matplotlib.pyplot as plt


class AttentionXCiT(Xcit):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            global_pool='token',
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            pos_drop_rate=0.,
            proj_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            act_layer=None,
            norm_layer=None,
            cls_attn_layers=2,
            use_pos_embed=True,
            eta=1.,
            tokens_norm=False,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate after positional embedding, and in XCA/CA projection + MLP
            pos_drop_rate: position embedding dropout rate
            proj_drop_rate (float): projection dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate (constant across all layers)
            norm_layer: (nn.Module): normalization layer
            cls_attn_layers: (int) Depth of Class attention layers
            use_pos_embed: (bool) whether to use positional encoding
            eta: (float) layerscale initialization value
            tokens_norm: (bool) Whether to normalize all tokens or just the cls_token in the CA

        Notes:
            - Although `layer_norm` is user specifiable, there are hard-coded `BatchNorm2d`s in the local patch
              interaction (class LPI) and the patch embedding (class ConvPatchEmbed)
        """
        super().__init__(img_size=img_size,
                         patch_size=patch_size,
                         in_chans=in_chans,
                         num_classes=num_classes,
                         global_pool=global_pool,
                         embed_dim=embed_dim,
                         depth=depth,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias,
                         drop_rate=drop_rate,
                         pos_drop_rate=pos_drop_rate,
                         attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate,
                         act_layer=act_layer,
                         norm_layer=norm_layer,
                         cls_attn_layers=cls_attn_layers,
                         use_pos_embed=use_pos_embed,
                         eta=eta,
                         tokens_norm=tokens_norm)
        self.patch_size = patch_size
        assert global_pool in ('', 'avg', 'token')
        img_size = to_2tuple(img_size)
        assert (img_size[0] % patch_size == 0) and (img_size[0] % patch_size == 0), \
            '`patch_size` should divide image dimensions evenly'
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.cls_attn_blocks = nn.ModuleList([
            AttentionClassAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                act_layer=act_layer,
                norm_layer=norm_layer,
                eta=eta,
                tokens_norm=tokens_norm,
            )
            for _ in range(cls_attn_layers)])

    def forward_features(self, x):
        B = x.shape[0]
        # x is (B, N, C). (Hp, Hw) is (height in units of patches, width in units of patches)
        x, (Hp, Wp) = self.patch_embed(x)

        if self.pos_embed is not None:
            # `pos_embed` (B, C, Hp, Wp), reshape -> (B, C, N), permute -> (B, N, C)
            pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding
        x = self.pos_drop(x)

        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, Hp, Wp)
            else:
                x = blk(x, Hp, Wp)

        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)

        for blk in self.cls_attn_blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x, attn = checkpoint(blk, x)
            else:
                x, attn = blk(x)

        x = self.norm(x)
        return x, attn

    def forward(self, x):
        x, _ = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def attention_forward(self, x):
        width, height = x.shape[-2:]
        width = width // self.patch_size
        height = height // self.patch_size

        x, attn = self.forward_features(x)

        # n_cls_tkns is the cls_tkns concatenated in forward_features
        n_cls_tkns = 1
        attn = attn[..., n_cls_tkns:].view(*attn.shape[:-1], width, height).squeeze(dim=2)

        x = self.forward_head(x)
        return x, attn


class AttentionClassAttentionBlock(ClassAttentionBlock):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 proj_drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 eta=1.,
                 tokens_norm=False, ):
        super().__init__(dim=dim,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias,
                         proj_drop=proj_drop,
                         attn_drop=attn_drop,
                         drop_path=drop_path,
                         act_layer=act_layer,
                         norm_layer=norm_layer,
                         eta=eta,
                         tokens_norm=tokens_norm)
        self.attn = AttentionClassAttn(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)

    def forward(self, x):
        x_norm1 = self.norm1(x)
        x_attn, attn = self.attn(x_norm1)
        x_attn = torch.cat([x_attn, x_norm1[:, 1:]], dim=1)
        x = x + self.drop_path(self.gamma1 * x_attn)
        if self.tokens_norm:
            x = self.norm2(x)
        else:
            x = torch.cat([self.norm2(x[:, 0:1]), x[:, 1:]], dim=1)
        x_res = x
        cls_token = x[:, 0:1]
        cls_token = self.gamma2 * self.mlp(cls_token)
        x = torch.cat([cls_token, x[:, 1:]], dim=1)
        x = x_res + self.drop_path(x)
        return x, attn


class AttentionClassAttn(ClassAttn):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        raw_attn = q @ k.transpose(-2, -1)
        attn = raw_attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_cls = attn @ v

        x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls, raw_attn


def checkpoint_filter_fn(state_dict, model):
    if 'model' in state_dict:
        state_dict = state_dict['model']
    # For consistency with timm's transformer models while being compatible with official weights source we rename
    # pos_embeder to pos_embed. Also account for use_pos_embed == False
    use_pos_embed = getattr(model, 'pos_embed', None) is not None
    pos_embed_keys = [k for k in state_dict if k.startswith('pos_embed')]
    for k in pos_embed_keys:
        if use_pos_embed:
            state_dict[k.replace('pos_embeder.', 'pos_embed.')] = state_dict.pop(k)
        else:
            del state_dict[k]
    # timm's implementation of class attention in CaiT is slightly more efficient as it does not compute query vectors
    # for all tokens, just the class token. To use official weights source we must split qkv into q, k, v
    if 'cls_attn_blocks.0.attn.qkv.weight' in state_dict and 'cls_attn_blocks.0.attn.q.weight' in model.state_dict():
        num_ca_blocks = len(model.cls_attn_blocks)
        for i in range(num_ca_blocks):
            qkv_weight = state_dict.pop(f'cls_attn_blocks.{i}.attn.qkv.weight')
            qkv_weight = qkv_weight.reshape(3, -1, qkv_weight.shape[-1])
            for j, subscript in enumerate('qkv'):
                state_dict[f'cls_attn_blocks.{i}.attn.{subscript}.weight'] = qkv_weight[j]
            qkv_bias = state_dict.pop(f'cls_attn_blocks.{i}.attn.qkv.bias', None)
            if qkv_bias is not None:
                qkv_bias = qkv_bias.reshape(3, -1)
                for j, subscript in enumerate('qkv'):
                    state_dict[f'cls_attn_blocks.{i}.attn.{subscript}.bias'] = qkv_bias[j]
    return state_dict


def _create_xcit(variant, pretrained=False, default_cfg=None, **kwargs):
    model = build_model_with_cfg(
        AttentionXCiT,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs,
    )
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj.0.0', 'classifier': 'head',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    # Patch size 16
    'xcit_nano_12_p16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_nano_12_p16_224.pth'),
    'xcit_nano_12_p16_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_nano_12_p16_224_dist.pth'),
    'xcit_nano_12_p16_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_nano_12_p16_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_tiny_12_p16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_12_p16_224.pth'),
    'xcit_tiny_12_p16_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_12_p16_224_dist.pth'),
    'xcit_tiny_12_p16_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_12_p16_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_tiny_24_p16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_24_p16_224.pth'),
    'xcit_tiny_24_p16_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_24_p16_224_dist.pth'),
    'xcit_tiny_24_p16_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_24_p16_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_small_12_p16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p16_224.pth'),
    'xcit_small_12_p16_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p16_224_dist.pth'),
    'xcit_small_12_p16_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p16_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_small_24_p16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_24_p16_224.pth'),
    'xcit_small_24_p16_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_24_p16_224_dist.pth'),
    'xcit_small_24_p16_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_24_p16_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_medium_24_p16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_medium_24_p16_224.pth'),
    'xcit_medium_24_p16_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_medium_24_p16_224_dist.pth'),
    'xcit_medium_24_p16_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_medium_24_p16_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_large_24_p16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_large_24_p16_224.pth'),
    'xcit_large_24_p16_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_large_24_p16_224_dist.pth'),
    'xcit_large_24_p16_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_large_24_p16_384_dist.pth', input_size=(3, 384, 384)),

    # Patch size 8
    'xcit_nano_12_p8_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_nano_12_p8_224.pth'),
    'xcit_nano_12_p8_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_nano_12_p8_224_dist.pth'),
    'xcit_nano_12_p8_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_nano_12_p8_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_tiny_12_p8_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_12_p8_224.pth'),
    'xcit_tiny_12_p8_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_12_p8_224_dist.pth'),
    'xcit_tiny_12_p8_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_12_p8_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_tiny_24_p8_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_24_p8_224.pth'),
    'xcit_tiny_24_p8_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_24_p8_224_dist.pth'),
    'xcit_tiny_24_p8_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_24_p8_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_small_12_p8_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p8_224.pth'),
    'xcit_small_12_p8_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p8_224_dist.pth'),
    'xcit_small_12_p8_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p8_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_small_24_p8_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_24_p8_224.pth'),
    'xcit_small_24_p8_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_24_p8_224_dist.pth'),
    'xcit_small_24_p8_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_24_p8_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_medium_24_p8_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_medium_24_p8_224.pth'),
    'xcit_medium_24_p8_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_medium_24_p8_224_dist.pth'),
    'xcit_medium_24_p8_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_medium_24_p8_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_large_24_p8_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_large_24_p8_224.pth'),
    'xcit_large_24_p8_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_large_24_p8_224_dist.pth'),
    'xcit_large_24_p8_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_large_24_p8_384_dist.pth', input_size=(3, 384, 384)),
})


@register_model
def xcit_nano_12_p16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=128, depth=12, num_heads=4, eta=1.0, tokens_norm=False)
    model = _create_xcit('xcit_nano_12_p16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_nano_12_p16_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=128, depth=12, num_heads=4, eta=1.0, tokens_norm=False, img_size=384)
    model = _create_xcit('xcit_nano_12_p16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_tiny_12_p16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, eta=1.0, tokens_norm=True)
    model = _create_xcit('xcit_tiny_12_p16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_tiny_12_p16_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, eta=1.0, tokens_norm=True)
    model = _create_xcit('xcit_tiny_12_p16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_small_12_p16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=8, eta=1.0, tokens_norm=True)
    model = _create_xcit('xcit_small_12_p16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_small_12_p16_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=8, eta=1.0, tokens_norm=True)
    model = _create_xcit('xcit_small_12_p16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_tiny_24_p16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=192, depth=24, num_heads=4, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_tiny_24_p16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_tiny_24_p16_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=192, depth=24, num_heads=4, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_tiny_24_p16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_small_24_p16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=384, depth=24, num_heads=8, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_small_24_p16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_small_24_p16_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=384, depth=24, num_heads=8, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_small_24_p16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_medium_24_p16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=512, depth=24, num_heads=8, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_medium_24_p16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_medium_24_p16_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=512, depth=24, num_heads=8, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_medium_24_p16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_large_24_p16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=768, depth=24, num_heads=16, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_large_24_p16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_large_24_p16_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=768, depth=24, num_heads=16, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_large_24_p16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


# Patch size 8x8 models
@register_model
def xcit_nano_12_p8_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=128, depth=12, num_heads=4, eta=1.0, tokens_norm=False)
    model = _create_xcit('xcit_nano_12_p8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_nano_12_p8_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=128, depth=12, num_heads=4, eta=1.0, tokens_norm=False)
    model = _create_xcit('xcit_nano_12_p8_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_tiny_12_p8_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=192, depth=12, num_heads=4, eta=1.0, tokens_norm=True)
    model = _create_xcit('xcit_tiny_12_p8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_tiny_12_p8_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=192, depth=12, num_heads=4, eta=1.0, tokens_norm=True)
    model = _create_xcit('xcit_tiny_12_p8_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_small_12_p8_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=384, depth=12, num_heads=8, eta=1.0, tokens_norm=True)
    model = _create_xcit('xcit_small_12_p8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_small_12_p8_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=384, depth=12, num_heads=8, eta=1.0, tokens_norm=True)
    model = _create_xcit('xcit_small_12_p8_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_tiny_24_p8_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=192, depth=24, num_heads=4, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_tiny_24_p8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_tiny_24_p8_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=192, depth=24, num_heads=4, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_tiny_24_p8_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_small_24_p8_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=384, depth=24, num_heads=8, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_small_24_p8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_small_24_p8_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=384, depth=24, num_heads=8, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_small_24_p8_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_medium_24_p8_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=512, depth=24, num_heads=8, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_medium_24_p8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_medium_24_p8_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=512, depth=24, num_heads=8, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_medium_24_p8_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_large_24_p8_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=768, depth=24, num_heads=16, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_large_24_p8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_large_24_p8_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=768, depth=24, num_heads=16, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_large_24_p8_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


register_model_deprecations(__name__, {
    # Patch size 16
    'xcit_nano_12_p16_224_dist': 'xcit_nano_12_p16_224.fb_dist_in1k',
    'xcit_nano_12_p16_384_dist': 'xcit_nano_12_p16_384.fb_dist_in1k',
    'xcit_tiny_12_p16_224_dist': 'xcit_tiny_12_p16_224.fb_dist_in1k',
    'xcit_tiny_12_p16_384_dist': 'xcit_tiny_12_p16_384.fb_dist_in1k',
    'xcit_tiny_24_p16_224_dist': 'xcit_tiny_24_p16_224.fb_dist_in1k',
    'xcit_tiny_24_p16_384_dist': 'xcit_tiny_24_p16_384.fb_dist_in1k',
    'xcit_small_12_p16_224_dist': 'xcit_small_12_p16_224.fb_dist_in1k',
    'xcit_small_12_p16_384_dist': 'xcit_small_12_p16_384.fb_dist_in1k',
    'xcit_small_24_p16_224_dist': 'xcit_small_24_p16_224.fb_dist_in1k',
    'xcit_medium_24_p16_224_dist': 'xcit_medium_24_p16_224.fb_dist_in1k',
    'xcit_medium_24_p16_384_dist': 'xcit_medium_24_p16_384.fb_dist_in1k',
    'xcit_large_24_p16_224_dist': 'xcit_large_24_p16_224.fb_dist_in1k',
    'xcit_large_24_p16_384_dist': 'xcit_large_24_p16_384.fb_dist_in1k',

    # Patch size 8
    'xcit_nano_12_p8_224_dist': 'xcit_nano_12_p8_224.fb_dist_in1k',
    'xcit_nano_12_p8_384_dist': 'xcit_nano_12_p8_384.fb_dist_in1k',
    'xcit_tiny_12_p8_224_dist': 'xcit_tiny_12_p8_224.fb_dist_in1k',
    'xcit_tiny_12_p8_384_dist': 'xcit_tiny_12_p8_384.fb_dist_in1k',
    'xcit_tiny_24_p8_224_dist': 'xcit_tiny_24_p8_224.fb_dist_in1k',
    'xcit_tiny_24_p8_384_dist': 'xcit_tiny_24_p8_384.fb_dist_in1k',
    'xcit_small_12_p8_224_dist': 'xcit_small_12_p8_224.fb_dist_in1k',
    'xcit_small_12_p8_384_dist': 'xcit_small_12_p8_384.fb_dist_in1k',
    'xcit_small_24_p8_224_dist': 'xcit_small_24_p8_224.fb_dist_in1k',
    'xcit_small_24_p8_384_dist': 'xcit_small_24_p8_384.fb_dist_in1k',
    'xcit_medium_24_p8_224_dist': 'xcit_medium_24_p8_224.fb_dist_in1k',
    'xcit_medium_24_p8_384_dist': 'xcit_medium_24_p8_384.fb_dist_in1k',
    'xcit_large_24_p8_224_dist': 'xcit_large_24_p8_224.fb_dist_in1k',
    'xcit_large_24_p8_384_dist': 'xcit_large_24_p8_384.fb_dist_in1k',
})

if __name__ == '__main__':
    model = xcit_small_12_p8_224(pretrained=True)
    inp = torch.randn((1, 3, 224, 224))
    x, attn = model.attention_forward(inp)
    attn = attn.transpose(1,0).squeeze(dim=2).detach().numpy()
    for attn_head in attn:
        for batch_head in attn_head:
            plt.imshow(batch_head, interpolation='nearest')
            plt.show()
            plt.close()
    print('x')
