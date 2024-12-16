import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from detectron2.utils import comm

import open_clip
from scsd.clip_surgery import clip

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


GENERAL_DOMAIN_PROMPTS = [
    "a photo of a {}.",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "a photo of a {} in the scene.",
    "There is a {} in the scene.",
    "There is the {} in the scene.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]

SPECIFIC_DOMAIN_PROMPTS = [
    "a photo of a {}{}.",
    "a photo of a small {}{}.",
    "a photo of a medium {}{}.",
    "a photo of a large {}{}.",
    "This is a photo of a {}{}.",
    "This is a photo of a small {}{}.",
    "This is a photo of a medium {}{}.",
    "This is a photo of a large {}{}.",
    "a photo of a {} in the scene{}.",
    "There is a {} in the scene{}.",
    "There is the {} in the scene{}.",
    "There is a small {} in the scene{}.",
    "There is a medium {} in the scene{}.",
    "There is a large {} in the scene{}.",
]

# For Synthetic
CONDITIONAL_DOMAIN_PROMPTS = ["", " in night", " in fog", " in rain", " in snow", "in city"]


@BACKBONE_REGISTRY.register()
class CLIP(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        model_name = cfg.MODEL.CLIP.CLIP_MODEL_NAME
        width = cfg.MODEL.FC_CLIP.EMBED_DIM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = clip.load(model_name, device=device)
        self.device = device
        self.clip_model = clip_model.float()
        self.model_name = model_name
        self.cache = None

        model_name = model_name.lower()
        print(clip_model, model_name)
        if 'vit' in model_name:
            self.model_type = 'vit'
            if 'b-16' in model_name:
                self.output_channels = [768, 768, 768, 768, 768]
                self.out_indices = [3, 5, 7, 11]

        self._out_feature_strides = {
            # "stem": 2,
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
            "clip_embedding": -1
        }
        self._out_feature_channels = {
            # "stem": self.output_channels[0],
            "res2": self.output_channels[1],
            "res3": self.output_channels[2],
            "res4": self.output_channels[3],
            "res5": self.output_channels[4],
            "clip_embedding": self.dim_latent
        }

        self.freeze_everything()
        self.domain_bank = None
        self.init_domain_bank()

        scale = width ** -0.5
        self.spatial_size = (self.clip_model.visual.image_size[0] // self.clip_model.visual.patch_size[0], self.clip_model.visual.image_size[1] // self.clip_model.visual.patch_size[1])
        # self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.fpn_dim = width
        self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(self.fpn_dim),
                nn.GELU(),
                nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)      

    def init_domain_bank(self):
        con_domain_prompts = clip.tokenize(CONDITIONAL_DOMAIN_PROMPTS).to(self.device)
        con_domain_prompts = self.encode_text(con_domain_prompts)  # [num_vild_templates, D]
        con_domain_prompts /= con_domain_prompts.norm(dim=-1, keepdim=True)
        self.domain_bank = nn.Parameter(con_domain_prompts.unsqueeze(1))

    def freeze_everything(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def text_global_pool(self, x, text, pool_type="argmax"):
        if pool_type == 'first':
            pooled, tokens = x[:, 0], x[:, 1:]
        elif pool_type == 'last':
            pooled, tokens = x[:, -1], x[:, :-1]
        elif pool_type == 'argmax':
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            assert text is not None
            pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
        else:
            pooled = tokens = x

        return pooled, tokens

    def style_diff_encode(self, images, targets, class_names_list, is_train=True):
        B, _, H, W = images.tensor.shape
        cast_dtype = images.tensor.dtype

        style_embeddings = []
        style_diff_embeddings = []
        domain_embeddings = []
        other_domain_embeddings = []

        if not is_train:  # 测试阶段
            return None, None, None, None

        for image, target in zip(images, targets):
            # 存储每个像素点对应的风格和风格差异特征
            style_embedding = torch.zeros((H, W, self.dim_latent), dtype=cast_dtype, device=self.device)
            style_diff_embedding = torch.zeros((H, W, self.dim_latent), dtype=cast_dtype, device=self.device)

            # 存储每个像素点对应的类别
            label_masks = torch.zeros((H, W), dtype=torch.int64, device=self.device)
            # 将对应为True的位置填入类别索引
            for gt_classes, gt_masks in zip(target["labels"], target["masks"]):
                label_masks[gt_masks] = gt_classes

            # 取出当前mask中存在的类别名
            class_names = [class_names_list[i.item()] for i in target["labels"]]

            if len(class_names) == 0:
                style_embeddings.append(style_embedding)  # [H, W, D]
                style_diff_embeddings.append(style_diff_embedding)  # [H, W, D]
                domain_embeddings.append(self.domain_bank[0])  # [1, D]
                other_domain_embeddings.append(torch.cat([self.domain_bank[i] for i in range(1, len(CONDITIONAL_DOMAIN_PROMPTS))], dim=0))
            else:
                idx = random.randint(0, len(CONDITIONAL_DOMAIN_PROMPTS) - 1)
                for i, classname in enumerate(class_names):
                    specific_domain_prompts = [template.format(classname, CONDITIONAL_DOMAIN_PROMPTS[idx]) for template in SPECIFIC_DOMAIN_PROMPTS]
                    specific_domain_prompts = clip.tokenize(specific_domain_prompts).to(self.device)
                    specific_domain_embeds = self.encode_text(specific_domain_prompts)  # [num_domain_templates, D]
                    specific_domain_embeds /= specific_domain_embeds.norm(dim=-1, keepdim=True)
                    specific_domain_embeds = specific_domain_embeds.mean(dim=0)
                    specific_domain_embeds /= specific_domain_embeds.norm()

                    general_domain_prompts = [template.format(classname) for template in GENERAL_DOMAIN_PROMPTS]
                    general_domain_prompts = clip.tokenize(general_domain_prompts).to(self.device)
                    general_domain_embeds = self.encode_text(general_domain_prompts)  # [num_vild_templates, D]
                    general_domain_embeds /= general_domain_embeds.norm(dim=-1, keepdim=True)
                    general_domain_embeds = general_domain_embeds.mean(dim=0)
                    general_domain_embeds /= general_domain_embeds.norm()

                    domain_diff_embed = specific_domain_embeds - general_domain_embeds
                    style_embedding[label_masks == i] = specific_domain_embeds.float()
                    style_diff_embedding[label_masks == i] = domain_diff_embed.float()

                style_embeddings.append(style_embedding)
                style_diff_embeddings.append(style_diff_embedding)
                domain_embeddings.append(self.domain_bank[idx])  # [1, D]
                other_domain_embedding = [self.domain_bank[i] for i in range(len(CONDITIONAL_DOMAIN_PROMPTS)) if i != idx]
                other_domain_embeddings.append(torch.cat(other_domain_embedding, dim=0))  # [other_num_domain, D]

        style_embeddings = torch.stack(style_embeddings, dim=0).permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]
        style_diff_embeddings = torch.stack(style_diff_embeddings, dim=0).permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]
        domain_embeddings = torch.cat(domain_embeddings, dim=0)  # [B, D]
        other_domain_embeddings = torch.stack(other_domain_embeddings, dim=0)  # [B, other_num_domain, D]
        return style_embeddings, style_diff_embeddings, domain_embeddings, other_domain_embeddings

    @torch.no_grad()
    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.clip_model.dtype

        x = self.clip_model.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        # B = x.shape[0]
        x = x + self.clip_model.positional_embedding.to(cast_dtype)
        # x = torch.cat([self.prompt_dropout(self.prompt_proj(self.domain_aware).expand(x.shape[0], -1, -1)), x], dim=1)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x, _ = self.text_global_pool(x, text, pool_type="argmax")

        if self.clip_model.text_projection is not None:
            if isinstance(self.clip_model.text_projection, nn.Linear):
                x = self.clip_model.text_projection(x)
            else:
                x = x @ self.clip_model.text_projection

        return F.normalize(x, dim=-1) if normalize else x

    def extract_features(self, x):
        return {
            'vit': self.extract_features_vit,
        }[self.model_type](x)

    def _expand_token(self, token, batch_size: int):
        return token.view(1, 1, -1).expand(batch_size, -1, -1)

    def extract_features_vit(self, x):
        out = {}
        x = self.clip_model.visual.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)
        x = torch.cat([self._expand_token(self.clip_model.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)

        pos = self.clip_model.visual.positional_embedding.to(x.dtype)
        cls_pos = pos[0,:]
        spatial_pos = F.interpolate(pos[1:,].reshape(1, self.spatial_size[0], self.spatial_size[1], C).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H*W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)

        x = x + pos
        x = self.clip_model.visual.patch_dropout(x)
        x = self.clip_model.visual.ln_pre(x)

        features = []
        for i, blk in enumerate(self.clip_model.visual.transformer.resblocks):
            x = blk(x)
            if i in self.out_indices:
                xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W)
                features.append(xp.contiguous())

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            out[f'res{i+2}'] = ops[i](features[i])
        
        x = self.clip_model.visual.ln_post(x)
        x = x @ self.clip_model.visual.proj

        out['clip_pool_dense'] = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2) # B C H W
        out['clip_pool_global'] = x[:, :1]  # [B, D]

        return out

    def get_text_embeds(self, classnames):
        if self.cache is not None and not self.training:
            return self.cache

        text_features = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = [template.format(classname_splits[0]) for template in GENERAL_DOMAIN_PROMPTS]
            else:
                texts = [template.format(classname) for template in GENERAL_DOMAIN_PROMPTS]  # format with class
            texts = clip.tokenize(texts).to(self.device)
            class_embeddings = self.encode_text(texts)  # [num_templates, D]
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)
        text_features = torch.stack(text_features, dim=1).to(self.device).t()
        if not self.training:
            self.cache = text_features

        return text_features

    def forward(self, x):
        return self.extract_features(x)
    
    @property
    def dim_latent(self):
        return self.clip_model.text_projection.shape[-1]
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in ["res2", "res3", "res4", "res5", "clip_embedding"]
        }

    @property
    def size_divisibility(self):
        return -1