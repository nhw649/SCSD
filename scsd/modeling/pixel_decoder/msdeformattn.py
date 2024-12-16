# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.cuda.amp import autocast
import torch.fft as fft

# visualization
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.position_encoding import PositionEmbeddingSine
from ..transformer_decoder.transformer import _get_clones, _get_activation_fn
from ..transformer_decoder.mask2former_transformer_decoder import CrossAttentionLayer
from .ops.modules import MSDeformAttn


# MSDeformAttn Transformer encoder in deformable detr
class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4,
        ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


@SEM_SEG_HEADS_REGISTRY.register()
class MSDeformAttnPixelDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        # deformable transformer encoder args
        transformer_in_features: List[str],
        common_stride: int,
        clip_embed_dim,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }

        # this is the input shape of pixel decoder
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]
        
        # this is the input shape of transformer encoder (could use less features than pixel decoder
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]  # starting from "res2" to "res5"
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]  # to decide extra FPN layers

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res2)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.mask_features)
        
        self.maskformer_num_feature_levels = 3  # always use 3 scales
        self.common_stride = common_stride

        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)

            lateral_conv = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.clip_embed_dim = clip_embed_dim
        self.style_adapter = nn.Conv2d(clip_embed_dim, conv_dim, kernel_size=1, bias=False)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        # ret["transformer_dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["transformer_dim_feedforward"] = 1024  # use 1024 for deformable transformer encoder
        ret[
            "transformer_enc_layers"
        ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
        ret["transformer_in_features"] = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
        ret["common_stride"] = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        ret["clip_embed_dim"] = cfg.MODEL.CLIP.EMBED_DIM
        return ret

    def save_phase_as_image(self, feature_maps, output_dir, batched_inputs, f):
        B, D, H, W = feature_maps.shape
        os.makedirs(output_dir, exist_ok=True)

        for b in range(B):
            file_name = batched_inputs[b]["file_name"].split("/")[-1].split(".")[0]
            if file_name != "00458":
                continue
            origin_img = batched_inputs[b]["image"]
            origin_height, origin_width = origin_img.shape[-2], origin_img.shape[-1]
            # # 在通道维度取平均
            averaged_feature_map = feature_maps[b].mean(dim=0)

            min_val = averaged_feature_map.min().item()
            max_val = averaged_feature_map.max().item()
            if max_val - min_val < 1e-5:  # 如果值范围太小，调整范围
                min_val = 0
                max_val = 1
            
            # # 将特征图归一化到0-255范围内
            # normalized_feature_map = 255 * (averaged_feature_map - averaged_feature_map.min()) / (averaged_feature_map.max() - averaged_feature_map.min())
            # normalized_feature_map = normalized_feature_map.to(torch.uint8)
            # print(normalized_feature_map)
            # 将特征图归一化到0-255范围内，并增加偏移量
            normalized_feature_map = 255 * (averaged_feature_map - min_val) / (max_val - min_val)
            if output_dir == "source_amp_shifted":
                normalized_feature_map = normalized_feature_map.clamp(0, 255) + 50  # 增加偏移量
            normalized_feature_map = normalized_feature_map.clamp(0, 255)  # 再次限制在0-255范围内
            normalized_feature_map = normalized_feature_map.to(torch.uint8)
            # 将tensor转换为PIL图像
            feature_map_image = Image.fromarray(normalized_feature_map.cpu().numpy())

            # 调整大小到原始图像尺寸
            feature_map_image = feature_map_image.resize((origin_width, origin_height), Image.BILINEAR)

            # 保存图像
            output_path = os.path.join(output_dir, f"{file_name}_{f}_avg.png")
            feature_map_image.save(output_path)

            for d in range(5):
                # feature_map_image = Image.fromarray(feature_maps[b, d].detach().cpu().numpy())
                phase_img = feature_maps[b, d]
                phase_img_normalized = ((phase_img - phase_img.min()) / (phase_img.max() - phase_img.min()) * 255).to(torch.uint8)
                phase_img_normalized = Image.fromarray(phase_img_normalized.detach().cpu().numpy())
                img_resized = phase_img_normalized.resize((origin_width, origin_height), Image.BILINEAR)
                output_path = os.path.join(output_dir, f"{file_name}_{f}_{d}.png")
                img_resized.save(output_path)
                

            # for d in range(D):
            #     # 归一化特征图到 [0, 255]
            #     feature_map = feature_maps[b, d].detach().cpu().numpy()
            #     feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
            #     feature_map = (feature_map * 255).astype(np.uint8)

            #     # 保存为图片
            #     img = Image.fromarray(feature_map)
            #     img_resized = img.resize((origin_width, origin_height), Image.BILINEAR)
            #     print(img_resized)
            #     img_resized.save(os.path.join(output_dir, f'{file_name}_batch_{b}_depth_{d}.png'))

    def save_tensor_as_image(self, feature_maps, output_dir, batched_inputs, f):
        B, D, H, W = feature_maps.shape
        os.makedirs(output_dir, exist_ok=True)

        for b in range(B):
            file_name = batched_inputs[b]["file_name"].split("/")[-1].split(".")[0]
            # print(file_name)
            if file_name != "00458":
                continue
            origin_img = batched_inputs[b]["image"]
            origin_height, origin_width = origin_img.shape[-2], origin_img.shape[-1]
            # # 在通道维度取平均
            averaged_feature_map = feature_maps[b].mean(dim=0)

            min_val = averaged_feature_map.min().item()
            max_val = averaged_feature_map.max().item()
            if max_val - min_val < 1e-5:  # 如果值范围太小，调整范围
                min_val = 0
                max_val = 1
            
            # # 将特征图归一化到0-255范围内
            normalized_feature_map = 255 * (averaged_feature_map - averaged_feature_map.min()) / (averaged_feature_map.max() - averaged_feature_map.min())
            normalized_feature_map = normalized_feature_map.to(torch.uint8)
            # print(normalized_feature_map)
            # 将特征图归一化到0-255范围内，并增加偏移量
            # normalized_feature_map = 255 * (averaged_feature_map - min_val) / (max_val - min_val)
            # if output_dir == "source_amp_shifted":
            #     normalized_feature_map = normalized_feature_map.clamp(0, 255)  # 增加偏移量
            # normalized_feature_map = normalized_feature_map.clamp(0, 255)  # 再次限制在0-255范围内
            # normalized_feature_map = normalized_feature_map.to(torch.uint8)
            # 将tensor转换为PIL图像
            feature_map_image = Image.fromarray(normalized_feature_map.detach().cpu().numpy())

            # 调整大小到原始图像尺寸
            feature_map_image = feature_map_image.resize((origin_width, origin_height), Image.BILINEAR)

            # 保存图像
            output_path = os.path.join(output_dir, f"{file_name}_{f}_avg.png")
            feature_map_image.save(output_path)

            for d in range(5):
                # feature_map_image = Image.fromarray(feature_maps[b, d].detach().cpu().numpy())
                phase_img = feature_maps[b, d]
                phase_img_normalized = ((phase_img - phase_img.min()) / (phase_img.max() - phase_img.min()) * 255).to(torch.uint8)
                phase_img_normalized = Image.fromarray(phase_img_normalized.detach().cpu().numpy())
                img_resized = phase_img_normalized.resize((origin_width, origin_height), Image.BILINEAR)
                output_path = os.path.join(output_dir, f"{file_name}_{f}_{d}.png")
                img_resized.save(output_path)
            # for d in range(D):
            #     # 归一化特征图到 [0, 255]
            #     feature_map = feature_maps[b, d].detach().cpu().numpy()
            #     feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
            #     feature_map = (feature_map * 255).astype(np.uint8)

            #     # 保存为图片
            #     img = Image.fromarray(feature_map)
            #     img_resized = img.resize((origin_width, origin_height), Image.BILINEAR)
            #     print(img_resized)
            #     img_resized.save(os.path.join(output_dir, f'{file_name}_batch_{b}_depth_{d}.png'))

    def style_transformation(self, image_features, style_diff_embeddings, batched_inputs, f, is_all_zero=False, beta=1, low_freq_ratio=0.15):
        style_diff_features = self.style_adapter(style_diff_embeddings)
        style_diff_features = F.interpolate(style_diff_features, size=(image_features.shape[-2], image_features.shape[-1]), mode='bilinear')

        # 计算傅里叶变换
        source_fft = fft.fft2(image_features, dim=(-2, -1))

        # 分离幅度和相位
        source_amp, source_phase = torch.abs(source_fft), torch.angle(source_fft)
        # self.save_tensor_as_image(source_phase, f'source_phase', batched_inputs, f)

        # 使用 fftshift 将频谱数据重新排列，使得零频率分量位于数组的中心位置
        source_amp_shifted = fft.fftshift(source_amp)
        # self.save_tensor_as_image(source_amp_shifted, f'source_amp_shifted', batched_inputs, f)

        # 获取图像的尺寸
        B, C, H, W = image_features.shape

        # 计算低频区域的尺寸
        low_freq_h = int(H * low_freq_ratio)
        low_freq_w = int(W * low_freq_ratio)

        # 构建低频掩码
        low_freq_mask = torch.zeros_like(source_amp_shifted)
        center_h = H // 2
        center_w = W // 2
        low_freq_mask[:, :, center_h - low_freq_h//2:center_h + low_freq_h//2, center_w - low_freq_w//2:center_w + low_freq_w//2] = 1

        # 分离低频部分
        low_freq_amp = source_amp_shifted * low_freq_mask
        # self.save_tensor_as_image(low_freq_amp, f'low_freq_amp', batched_inputs, f)

        # 变换低频部分的幅度
        transformed_low_freq_amp = low_freq_amp * (1 + beta * torch.tanh(style_diff_features))
        # self.save_tensor_as_image(transformed_low_freq_amp, f'transformed_low_freq_amp', batched_inputs, f)
        
        # 将变换后的低频部分和高频部分合并
        transformed_amp_shifted = source_amp_shifted.clone()
        transformed_amp_shifted[low_freq_mask == 1] = transformed_low_freq_amp[low_freq_mask == 1]
        # self.save_tensor_as_image(transformed_amp_shifted, f'transformed_amp_shifted', batched_inputs, f)

        # 使用 ifftshift 将频谱数据重新排列回原始位置
        transformed_amp = fft.ifftshift(transformed_amp_shifted)
        # self.save_tensor_as_image(transformed_amp, f'transformed_amp', batched_inputs, f)

        # 混合新的幅度和原始相位
        transformed_fft = torch.polar(transformed_amp, source_phase)

        # 计算逆傅里叶变换
        transformed_features = fft.ifft2(transformed_fft, dim=(-2, -1)).real  # [B, C, H, W]

        return transformed_features

    @autocast(enabled=False)
    def forward_features(self, features, batched_inputs):
        style_diff_embeddings = features["style_diff_embeddings"]

        guide_strength = [1.0, 2.0, 4.0]
        srcs = []
        pos = []
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision
            if self.training:
                x = self.input_proj[idx](x)
                image_domain_embeddings = self.style_transformation(x, style_diff_embeddings, batched_inputs, f, beta=guide_strength[idx])
                srcs.append(image_domain_embeddings)
            else:
                srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))

        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]

        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = output_conv(y)
            out.append(y)

        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        return self.mask_features(out[-1]), out[0], multi_scale_features, srcs
