import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange
from typing import Optional, Any, Dict, List

from .distributions import DiagonalGaussianDistribution
from .config import Config, AttnMode


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        print(f"building AttnBlock (vanilla) with {in_channels} in_channels")

        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class MemoryEfficientAttnBlock(nn.Module):
    """
    Uses xformers efficient implementation,
    see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    Note: this is a single-head self-attention operation
    """

    #
    def __init__(self, in_channels):
        super().__init__()
        print(
            f"building MemoryEfficientAttnBlock (xformers) with {in_channels} in_channels"
        )
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.attention_op: Optional[Any] = None

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, C, H, W = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b (h w) c"), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], 1, C)
            .permute(0, 2, 1, 3)
            .reshape(B * 1, t.shape[1], C)
            .contiguous(),
            (q, k, v),
        )
        out = Config.xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        out = (
            out.unsqueeze(0)
            .reshape(B, 1, out.shape[1], C)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], C)
        )
        out = rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)
        out = self.proj_out(out)
        return x + out


class SDPAttnBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        print(f"building SDPAttnBlock (sdp) with {in_channels} in_channels")
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, C, H, W = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b (h w) c"), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], 1, C)
            .permute(0, 2, 1, 3)
            .reshape(B * 1, t.shape[1], C)
            .contiguous(),
            (q, k, v),
        )
        out = F.scaled_dot_product_attention(q, k, v)

        out = (
            out.unsqueeze(0)
            .reshape(B, 1, out.shape[1], C)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], C)
        )
        out = rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)
        out = self.proj_out(out)
        return x + out


def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    assert attn_type in [
        "vanilla",
        "sdp",
        "xformers",
        "linear",
        "none",
    ], f"attn_type {attn_type} unknown"
    if attn_type == "vanilla":
        assert attn_kwargs is None
        return AttnBlock(in_channels)
    elif attn_type == "sdp":
        return SDPAttnBlock(in_channels)
    elif attn_type == "xformers":
        return MemoryEfficientAttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        raise NotImplementedError()


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        use_linear_attn=False,
        **ignore_kwargs,
    ):
        super().__init__()
        ### setup attention type
        if Config.attn_mode == AttnMode.SDP:
            attn_type = "sdp"
        elif Config.attn_mode == AttnMode.XFORMERS:
            attn_type = "xformers"
        else:
            attn_type = "vanilla"
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class LayerNorm2d(nn.Module):
    """2D LayerNorm，适用于图像特征 (B, C, H, W)"""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SimilarityAwareFusion(nn.Module):
    """相似度感知的特征融合模块，根据lq特征和检索特征的相似度动态调整融合权重"""
    
    def __init__(self, channels: int, similarity_threshold: float = 0.3):
        super().__init__()
        self.channels = channels
        self.similarity_threshold = similarity_threshold
        
        # 特征投影：将特征映射到共享空间
        self.lq_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.ref_proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        # 局部相似度计算：使用卷积核计算局部相似度
        self.similarity_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid()  # 输出 ∈ (0, 1)
        )
        
        # 全局相似度作为补充
        self.global_similarity = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels * 2, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, 1),
            nn.Sigmoid()
        )
        
        # 最终融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
        
    def forward(self, h_dec: torch.Tensor, f_lq: torch.Tensor, f_ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_dec: decoder特征 (B, C, H, W)
            f_lq: lq图像特征 (B, C, H, W) 
            f_ref: 检索到的特征 (B, C, H, W)
        Returns:
            fused_feat: 融合后的特征 (B, C, H, W)
        """
        # 空间对齐
        if f_lq.shape[2:] != h_dec.shape[2:]:
            f_lq = F.interpolate(f_lq, size=h_dec.shape[2:], mode='bilinear', align_corners=False)
        if f_ref.shape[2:] != h_dec.shape[2:]:
            f_ref = F.interpolate(f_ref, size=h_dec.shape[2:], mode='bilinear', align_corners=False)
        
        # 特征投影
        lq_emb = self.lq_proj(f_lq)
        ref_emb = self.ref_proj(f_ref)
        
        # 计算特征差异
        diff = lq_emb - ref_emb
        
        # 局部相似度：基于空间邻域计算
        local_sim_input = torch.cat([lq_emb, ref_emb], dim=1)
        local_similarity = self.similarity_conv(local_sim_input)  # (B, 1, H, W)
        
        # 全局相似度：作为局部相似度的补充
        global_sim_input = torch.cat([lq_emb, ref_emb], dim=1)
        global_similarity = self.global_similarity(global_sim_input)  # (B, 1)
        global_similarity = global_similarity.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        
        # 结合局部和全局相似度
        combined_similarity = 0.7 * local_similarity + 0.3 * global_similarity
        
        # 应用相似度阈值：低于阈值的区域使用更多lq特征
        similarity_weight = torch.clamp(combined_similarity, min=self.similarity_threshold, max=1.0)
        
        # 动态融合：根据相似度调整检索特征和lq特征的权重
        fused_feat = similarity_weight * ref_emb + (1 - similarity_weight) * lq_emb
        
        # 与decoder特征融合（残差形式）
        fusion_input = torch.cat([h_dec, fused_feat], dim=1)
        fusion_out = self.fusion(fusion_input)
        
        return h_dec + fusion_out


class ResidualGatedFusion(nn.Module):
    """轻量级残差门控融合模块，用于融合检索到的特征"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # 特征投影：将检索特征投影到当前特征空间
        self.feature_proj = nn.Conv2d(channels, channels, 1)
        
        # 门控机制：学习融合权重
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, lq_feat: torch.Tensor, retrieved_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lq_feat: 当前特征 (B, C, H, W)
            retrieved_feat: 检索到的特征 (B, C, H, W)
        Returns:
            fused_feat: 融合后的特征 (B, C, H, W)
        """
        # 投影检索特征以对齐特征空间
        proj_retrieved = self.feature_proj(retrieved_feat)
        
        # 计算门控权重
        gate = self.gate(torch.cat([lq_feat, proj_retrieved], dim=1))
        
        # 残差门控融合：当前特征 + 门控的检索特征
        fused_feat = lq_feat + gate * proj_retrieved
        
        return fused_feat


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        enable_feature_fusion=False,
        fusion_levels=None,
        **ignorekwargs,
    ):
        super().__init__()
        ### setup attention type
        if Config.attn_mode == AttnMode.SDP:
            attn_type = "sdp"
        elif Config.attn_mode == AttnMode.XFORMERS:
            attn_type = "xformers"
        else:
            attn_type = "vanilla"
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        
        # 特征融合相关参数
        self.enable_feature_fusion = enable_feature_fusion
        self.fusion_levels = fusion_levels if fusion_levels is not None else list(range(self.num_resolutions))

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # 特征融合模块
        if self.enable_feature_fusion:
            self.fusion_blocks = nn.ModuleList()
            for i_level in range(self.num_resolutions):
                if i_level in self.fusion_levels:
                    channels = ch * ch_mult[i_level]
                    print(f"创建相似度感知融合模块，channels: {channels}")
                    self.fusion_blocks.append(SimilarityAwareFusion(channels))
                else:
                    self.fusion_blocks.append(None)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z, retrieved_features=None, lq_features=None):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
                
                # 特征融合：在每个up block的第一个resblock后进行融合
                if (self.enable_feature_fusion and 
                    retrieved_features is not None and 
                    i_level in self.fusion_levels and
                    i_block == 0 and 
                    self.fusion_blocks[i_level] is not None):
                    
                    # 找到对应的特征索引
                    feature_idx = self.fusion_levels.index(i_level)
                    if feature_idx < len(retrieved_features):
                        # 确保特征图尺寸匹配
                        if h.shape[2:] == retrieved_features[feature_idx].shape[2:]:
                            # 获取对应的lq特征
                            lq_feat = None
                            if lq_features is not None and feature_idx < len(lq_features):
                                lq_feat = lq_features[feature_idx]
                            
                            # 使用新的ConvNeXt融合模块
                            if lq_feat is not None:
                                h = self.fusion_blocks[i_level](h, lq_feat, retrieved_features[feature_idx])
                            else:
                                # 如果没有lq特征，使用旧的融合方式（向后兼容）
                                h = self.fusion_blocks[i_level](h, h, retrieved_features[feature_idx])
                        
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class AutoencoderKL(nn.Module):

    def __init__(self, ddconfig, embed_dim, enable_feature_fusion=False, fusion_levels=None):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig, enable_feature_fusion=enable_feature_fusion, fusion_levels=fusion_levels)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        
        # 特征融合相关
        self.enable_feature_fusion = enable_feature_fusion
        self.fusion_levels = fusion_levels
        
        # 从decoder中获取fusion_blocks，使其在AutoencoderKL级别可访问
        if self.enable_feature_fusion and hasattr(self.decoder, 'fusion_blocks'):
            self.fusion_blocks = self.decoder.fusion_blocks
        else:
            self.fusion_blocks = None
        
        # 用于存储中间特征
        self.intermediate_features = {}
        self.feature_extractor = None
        if self.enable_feature_fusion:
            self._setup_feature_extraction()

    def _setup_feature_extraction(self):
        """设置特征提取钩子"""
        self.feature_extractor = {}
        
        def get_activation(name):
            def hook(module, input, output):
                self.intermediate_features[name] = output.detach()
            return hook
        
        # 为每个down block的最后一个resnet block注册钩子
        for i, down_block in enumerate(self.encoder.down):
            # 注册在最后一个resnet block上，这样可以获得该层级的最终特征
            last_block = down_block.block[-1]
            last_block.register_forward_hook(get_activation(f'down_{i}'))

    def encode(self, x, return_features=False):
        self.intermediate_features.clear()
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        
        if return_features:
            return posterior, self.intermediate_features.copy()
        return posterior

    def decode(self, z, retrieved_features=None, lq_features=None):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, retrieved_features, lq_features)
        return dec

    def forward(self, input, sample_posterior=True, retrieved_features=None, lq_features=None):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
            
        dec = self.decode(z, retrieved_features, lq_features)
        return dec, posterior
    
    def get_intermediate_features(self):
        """获取中间特征，用于特征库构建"""
        return self.intermediate_features.copy()

