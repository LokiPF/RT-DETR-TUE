"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math 
import copy 
import functools
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 
from typing import List, Optional

from .topological_uncertainty import (
    maximum_spanning_tree_signature,
    maximum_spanning_tree_signature_batch,
)
from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob

from ...core import register

__all__ = [
    "RTDETRTransformerv2TUE",
    "bbox_gaussian_nll",
]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x, return_layer_inputs=False):
        layer_inputs = []

        for i, layer in enumerate(self.layers):
            if return_layer_inputs:
                layer_inputs.append(x)

            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)

        if return_layer_inputs:
            return x, layer_inputs

        return x


class BBoxUncertaintyHead(nn.Module):
    """Predict normalized xyxy standard deviations for one decoder query."""

    def __init__(
        self,
        decoder_dim,
        hidden_dim=64,
        num_layers=2,
        min_std=1e-4,
        max_std=1.0,
        initial_std=0.05,
        act="relu",
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("bbox_uncertainty_num_layers must be at least 1.")
        if not 0 < min_std < initial_std < max_std:
            raise ValueError(
                "Require 0 < min_std < initial_std < max_std for the "
                "bbox uncertainty head."
            )

        # h + TU + confidence + log(area) + |decoder bbox refinement|
        input_dim = decoder_dim + 1 + 1 + 1 + 4
        self.decoder_feature_norm = nn.LayerNorm(decoder_dim)
        self.mlp = MLP(
            input_dim,
            hidden_dim,
            4,
            num_layers,
            act=act,
        )
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.initial_std = float(initial_std)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp.layers:
            init.xavier_uniform_(layer.weight)
            init.constant_(layer.bias, 0)

        # Start with a constant, useful scale. The head then learns departures
        # from it rather than beginning with arbitrary query-dependent widths.
        final_layer = self.mlp.layers[-1]
        init.constant_(final_layer.weight, 0)
        softplus_target = self.initial_std - self.min_std
        initial_bias = math.log(math.expm1(softplus_target))
        init.constant_(final_layer.bias, initial_bias)

    def forward(
        self,
        decoder_features,
        bbox_tu,
        confidence,
        log_area,
        abs_bbox_refinement,
    ):
        scalar_features = torch.cat(
            (
                bbox_tu.unsqueeze(-1),
                confidence.unsqueeze(-1),
                log_area.unsqueeze(-1),
                abs_bbox_refinement,
            ),
            dim=-1,
        )
        features = torch.cat(
            (
                self.decoder_feature_norm(decoder_features),
                scalar_features,
            ),
            dim=-1,
        )
        raw_std = self.mlp(features)
        std = F.softplus(raw_std) + self.min_std
        return std.clamp_max(self.max_std)


def bbox_gaussian_nll(
    pred_xyxy,
    target_xyxy,
    pred_std_xyxy,
    valid_mask: Optional[torch.Tensor] = None,
    reduction="mean",
):
    """Gaussian NLL for already matched, normalized xyxy boxes.

    Args:
        pred_xyxy: Tensor [..., 4] containing matched predicted box means.
        target_xyxy: Tensor [..., 4] containing the corresponding targets.
        pred_std_xyxy: Tensor [..., 4] from ``out["pred_bbox_std"]``.
        valid_mask: Optional boolean tensor with shape ``pred_xyxy.shape[:-1]``.
        reduction: ``"none"``, ``"mean"``, or ``"sum"``. ``"none"`` returns
            one summed four-coordinate loss per matched box.
    """
    if pred_xyxy.shape != target_xyxy.shape:
        raise ValueError("Predicted and target xyxy tensors must have equal shape.")
    if pred_xyxy.shape != pred_std_xyxy.shape or pred_xyxy.shape[-1] != 4:
        raise ValueError("BBox means and standard deviations must have shape [..., 4].")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"Unsupported reduction: {reduction}")

    finite_mask = (
        torch.isfinite(pred_xyxy).all(dim=-1)
        & torch.isfinite(target_xyxy).all(dim=-1)
        & torch.isfinite(pred_std_xyxy).all(dim=-1)
    )
    if valid_mask is not None:
        if valid_mask.shape != pred_xyxy.shape[:-1]:
            raise ValueError(
                "valid_mask must have shape pred_xyxy.shape[:-1]."
            )
        finite_mask = finite_mask & valid_mask.to(
            device=finite_mask.device,
            dtype=torch.bool,
        )

    coordinate_mask = finite_mask.unsqueeze(-1)
    safe_pred = torch.where(
        coordinate_mask,
        pred_xyxy,
        torch.zeros_like(pred_xyxy),
    )
    safe_target = torch.where(
        coordinate_mask,
        target_xyxy,
        torch.zeros_like(target_xyxy),
    )
    safe_std = torch.where(
        coordinate_mask,
        pred_std_xyxy,
        torch.ones_like(pred_std_xyxy),
    )
    std = safe_std.clamp_min(torch.finfo(pred_std_xyxy.dtype).eps)
    residual = safe_target - safe_pred
    coordinate_nll = (
        0.5 * (residual / std).square()
        + torch.log(std)
        + 0.5 * math.log(2.0 * math.pi)
    )
    per_box_nll = coordinate_nll.sum(dim=-1)

    if reduction == "none":
        return torch.where(
            finite_mask,
            per_box_nll,
            torch.full_like(per_box_nll, float("nan")),
        )

    valid_losses = per_box_nll[finite_mask]
    if valid_losses.numel() == 0:
        # Keep a differentiable zero when a batch has no matched valid boxes.
        return pred_std_xyxy.sum() * 0.0
    if reduction == "sum":
        return valid_losses.sum()
    return valid_losses.mean()


class MSDeformableAttention(nn.Module):
    def __init__(
        self, 
        embed_dim=256, 
        num_heads=8, 
        num_levels=4, 
        num_points=4, 
        method='default',
        offset_scale=0.5,
    ):
        """Multi-Scale Deformable Attention
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list
        
        num_points_scale = [1/n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method) 

        self._reset_parameters()

        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)


    def forward(self,
                query: torch.Tensor,
                reference_points: torch.Tensor,
                value: torch.Tensor,
                value_spatial_shapes: List[int],
                value_mask: torch.Tensor=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)

        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)

        output = self.output_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default'):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                target,
                reference_points,
                memory,
                memory_spatial_shapes,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(\
            self.with_pos_embed(target, query_pos_embed), 
            reference_points, 
            memory, 
            memory_spatial_shapes, 
            memory_mask)
        target = target + self.dropout2(target2)
        target = self.norm2(target)

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target)

        return target


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                target,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,
                collect_tu=False):
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        output = target

        tu_payload = None

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(
                output,
                ref_points_input,
                memory,
                memory_spatial_shapes,
                attn_mask,
                memory_mask,
                query_pos_embed,
            )

            collect_this_layer = (
                collect_tu
                and i == self.eval_idx
            )

            if collect_this_layer:
                bbox_delta, bbox_inputs = bbox_head[i](
                    output,
                    return_layer_inputs=True,
                )
            else:
                bbox_delta = bbox_head[i](output)

            inter_ref_bbox = torch.sigmoid(
                bbox_delta + inverse_sigmoid(ref_points_detach)
            )

            if collect_this_layer:
                tu_payload = {
                    # Input to final 256 -> 4 bbox layer:
                    "bbox_last_input": bbox_inputs[-1],

                    # Query embedding used by the classification and bbox heads.
                    "decoder_output": output,

                    # Reference before this layer's bbox refinement (cxcywh).
                    "previous_ref_bbox": ref_points_detach,

                    "decoder_layer": i,
                }

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()

        return (
                    torch.stack(dec_out_bboxes),
                    torch.stack(dec_out_logits),
                    tu_payload,
                )


@register()
class RTDETRTransformerv2TUE(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                num_classes=80,
                hidden_dim=256,
                num_queries=300,
                feat_channels=[512, 1024, 2048],
                feat_strides=[8, 16, 32],
                num_levels=3,
                num_points=4,
                nhead=8,
                num_layers=6,
                dim_feedforward=1024,
                dropout=0.,
                activation="relu",
                num_denoising=100,
                label_noise_ratio=0.5,
                box_noise_scale=1.0,
                learn_query_content=False,
                eval_spatial_size=None,
                eval_idx=-1,
                eps=1e-2, 
                aux_loss=True, 
                cross_attn_method='default', 
                query_select_method='default',
                tu_enabled=False,
                tu_prototype_path=None,
                tu_topk=50,
                tu_min_samples=20,
                bbox_uncertainty_enabled=False,
                bbox_uncertainty_input_mode="all",
                bbox_uncertainty_hidden_dim=64,
                bbox_uncertainty_num_layers=2,
                bbox_uncertainty_min_std=1e-4,
                bbox_uncertainty_max_std=1.0,
                bbox_uncertainty_initial_std=0.05,
                bbox_uncertainty_detach_inputs=True,
                bbox_uncertainty_train_tu_topk=0,):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss

        assert query_select_method in ('default', 'one2many', 'agnostic'), ''
        assert cross_attn_method in ('default', 'discrete'), ''
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_layers, eval_idx)
        if (
            bbox_uncertainty_enabled
            and self.decoder.eval_idx != num_layers - 1
        ):
            raise ValueError(
                "The bbox uncertainty head currently requires eval_idx to "
                "select the final decoder layer, so training and inference "
                "use the same query features."
            )

        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0: 
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)

        # if num_select_queries != self.num_queries:
        #     layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, activation='gelu')
        #     self.encoder = TransformerEncoder(layer, 1)

        self.enc_output = nn.Sequential(OrderedDict([
            ('proj', nn.Linear(hidden_dim, hidden_dim)),
            ('norm', nn.LayerNorm(hidden_dim,)),
        ]))

        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_layers)
        ])

        self.bbox_uncertainty_enabled = bbox_uncertainty_enabled
        valid_uncertainty_input_modes = (
            "all",
            "features_only",
            "tu_only",
        )
        if bbox_uncertainty_input_mode not in valid_uncertainty_input_modes:
            raise ValueError(
                "bbox_uncertainty_input_mode must be one of "
                f"{valid_uncertainty_input_modes}."
            )
        self.bbox_uncertainty_input_mode = bbox_uncertainty_input_mode
        self.bbox_uncertainty_use_tu = (
            bbox_uncertainty_input_mode != "features_only"
        )
        self.bbox_uncertainty_detach_inputs = (
            bbox_uncertainty_detach_inputs
        )
        self.bbox_uncertainty_train_tu_topk = int(
            bbox_uncertainty_train_tu_topk
        )
        if self.bbox_uncertainty_train_tu_topk < 0:
            raise ValueError(
                "bbox_uncertainty_train_tu_topk must be non-negative; "
                "zero means all queries."
            )

        if self.bbox_uncertainty_enabled:
            self.bbox_uncertainty_head = BBoxUncertaintyHead(
                decoder_dim=hidden_dim,
                hidden_dim=bbox_uncertainty_hidden_dim,
                num_layers=bbox_uncertainty_num_layers,
                min_std=bbox_uncertainty_min_std,
                max_std=bbox_uncertainty_max_std,
                initial_std=bbox_uncertainty_initial_std,
                act=activation,
            )
        else:
            self.bbox_uncertainty_head = None

        self.register_buffer(
            "bbox_tu_class_means",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "bbox_tu_class_valid",
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "bbox_tu_global_mean",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "bbox_tu_class_size_means",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "bbox_tu_class_size_valid",
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "bbox_tu_area_thresholds",
            torch.empty(0),
            persistent=False,
        )

        self.tu_enabled = tu_enabled
        self.tu_prototype_path = tu_prototype_path
        self.bbox_tu_topk = tu_topk
        self.tu_min_samples = tu_min_samples
        self.bbox_tu_loaded = False

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)

        self._reset_parameters()

        if self.tu_prototype_path is not None:
            self.load_tu_prototypes(self.tu_prototype_path)

    @torch.no_grad()
    def load_tu_prototypes(self, path):
        try:
            artifact = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            artifact = torch.load(path, map_location="cpu")

        metadata = artifact["metadata"]
        prototypes = artifact["prototypes"]

        eval_idx = self.decoder.eval_idx
        bbox_layer = self.dec_bbox_head[eval_idx].layers[-1]

        if metadata["decoder_eval_index"] != eval_idx:
            raise ValueError("Prototype decoder index does not match model.")

        if metadata["bbox_layer_in_features"] != bbox_layer.in_features:
            raise ValueError("Prototype input dimension does not match bbox layer.")

        if metadata["bbox_layer_out_features"] != bbox_layer.out_features:
            raise ValueError("Prototype output dimension does not match bbox layer.")

        signature_length = int(metadata["signature_length"])

        size_names = metadata.get(
            "size_bucket_names",
            ["small", "medium", "large"],
        )
        size_to_id = {
            name: index for index, name in enumerate(size_names)
        }
        num_size_buckets = len(size_names)

        class_size_means = torch.zeros(
            self.num_classes,
            num_size_buckets,
            signature_length,
            dtype=torch.float32,
        )
        class_size_valid = torch.zeros(
            self.num_classes,
            num_size_buckets,
            dtype=torch.bool,
        )

        class_means = torch.zeros(
            self.num_classes,
            signature_length,
            dtype=torch.float32,
        )
        class_valid = torch.zeros(
            self.num_classes,
            dtype=torch.bool,
        )

        # Keys look like "17:small".
        for key, value in prototypes.get("class_size", {}).items():
            class_text, size_name = key.split(":", maxsplit=1)

            class_id = int(class_text)
            size_id = size_to_id.get(size_name)
            count = int(value.get("count", 0))

            if (
                0 <= class_id < self.num_classes
                and size_id is not None
                and count >= self.tu_min_samples
            ):
                mean = value["mean"].float().flatten()

                if mean.numel() != signature_length:
                    raise ValueError(
                        f"Prototype {key} has signature length "
                        f"{mean.numel()}, expected {signature_length}."
                    )

                class_size_means[class_id, size_id] = mean
                class_size_valid[class_id, size_id] = True

        # Class-only fallback.
        for class_key, value in prototypes["class"].items():
            class_id = int(class_key)
            count = int(value.get("count", 0))

            if (
                0 <= class_id < self.num_classes
                and count >= self.tu_min_samples
            ):
                class_means[class_id] = value["mean"].float().flatten()
                class_valid[class_id] = True

        global_mean = prototypes["global"]["mean"].float().flatten()

        area_metadata = metadata["normalized_area_thresholds"]
        area_thresholds = torch.tensor(
            [
                float(area_metadata["small_max"]),
                float(area_metadata["medium_max"]),
            ],
            dtype=torch.float32,
        )

        device = bbox_layer.weight.device

        self.bbox_tu_class_size_means = class_size_means.to(device)
        self.bbox_tu_class_size_valid = class_size_valid.to(device)
        self.bbox_tu_class_means = class_means.to(device)
        self.bbox_tu_class_valid = class_valid.to(device)
        self.bbox_tu_global_mean = global_mean.to(device)
        self.bbox_tu_area_thresholds = area_thresholds.to(device)

        self.bbox_tu_loaded = True

        print(
            "[TU] loaded "
            f"{int(class_size_valid.sum())} valid class/size prototypes, "
            f"{int(class_valid.sum())} class fallbacks"
        )

    @torch.no_grad()
    def calculate_bbox_tu(
        self,
        activation,
        predicted_class,
        predicted_size,
    ):
        eval_idx = self.decoder.eval_idx
        weight = self.dec_bbox_head[eval_idx].layers[-1].weight

        signature = maximum_spanning_tree_signature(
            activation,
            weight,
            edge_score="abs_wx",
        ).to(weight.device)

        class_id = int(predicted_class)
        size_id = int(predicted_size)

        use_class_size = (
            0 <= class_id < self.num_classes
            and 0 <= size_id < self.bbox_tu_class_size_valid.shape[1]
            and bool(
                self.bbox_tu_class_size_valid[class_id, size_id].item()
            )
        )

        if use_class_size:
            mean_diagram = self.bbox_tu_class_size_means[
                class_id, size_id
            ]
        elif (
            0 <= class_id < self.num_classes
            and bool(self.bbox_tu_class_valid[class_id].item())
        ):
            mean_diagram = self.bbox_tu_class_means[class_id]
        else:
            mean_diagram = self.bbox_tu_global_mean

        return torch.sqrt(
            torch.mean((signature - mean_diagram) ** 2)
        )

    @torch.no_grad()
    def _calculate_bbox_tu_batch(
        self,
        logits,
        predicted_boxes,
        activations,
        topk,
    ):
        """Calculate TU for all queries, or only the top-k by confidence."""
        if not self.bbox_tu_loaded:
            raise RuntimeError(
                "BBox TU is required, but prototypes were not loaded. "
                "Set tu_prototype_path in the config or call "
                "load_tu_prototypes(path)."
            )

        if activations.shape[:2] != logits.shape[:2]:
            raise ValueError(
                "TU activation/query shape does not match decoder logits."
            )

        probabilities = logits.sigmoid()
        query_scores = probabilities.amax(dim=-1)
        query_classes = probabilities.argmax(dim=-1)

        predicted_areas = (
            predicted_boxes[..., 2].clamp_min(0)
            * predicted_boxes[..., 3].clamp_min(0)
        )
        small_max = self.bbox_tu_area_thresholds[0]
        medium_max = self.bbox_tu_area_thresholds[1]
        query_sizes = (
            (predicted_areas >= small_max).long()
            + (predicted_areas >= medium_max).long()
        )

        query_count = logits.shape[1]
        selected_mask = torch.ones(
            logits.shape[:2],
            dtype=torch.bool,
            device=logits.device,
        )
        if topk is not None and 0 < int(topk) < query_count:
            query_indices = query_scores.topk(
                int(topk),
                dim=1,
            ).indices
            selected_mask.zero_()
            selected_mask.scatter_(1, query_indices, True)

        bbox_tu = torch.full(
            logits.shape[:2],
            float("nan"),
            dtype=torch.float32,
            device=logits.device,
        )

        selected_activations = activations[selected_mask]
        selected_classes = query_classes[selected_mask]
        selected_sizes = query_sizes[selected_mask]

        eval_idx = self.decoder.eval_idx
        weight = self.dec_bbox_head[eval_idx].layers[-1].weight
        signatures = maximum_spanning_tree_signature_batch(
            selected_activations,
            weight,
            edge_score="abs_wx",
        )

        # Vectorized class/size -> class -> global prototype fallback.
        mean_diagrams = self.bbox_tu_global_mean.unsqueeze(0).expand(
            signatures.shape[0],
            -1,
        ).clone()

        class_valid = self.bbox_tu_class_valid[selected_classes]
        mean_diagrams[class_valid] = self.bbox_tu_class_means[
            selected_classes[class_valid]
        ]

        class_size_valid = self.bbox_tu_class_size_valid[
            selected_classes,
            selected_sizes,
        ]
        mean_diagrams[class_size_valid] = (
            self.bbox_tu_class_size_means[
                selected_classes[class_size_valid],
                selected_sizes[class_size_valid],
            ]
        )

        selected_tu = torch.sqrt(
            torch.mean(
                (signatures - mean_diagrams) ** 2,
                dim=-1,
            )
        )
        bbox_tu[selected_mask] = selected_tu

        return bbox_tu

    def _predict_bbox_uncertainty(
        self,
        logits,
        predicted_boxes,
        decoder_payload,
        bbox_tu=None,
    ):
        """Run the lightweight uncertainty head.

        ``predicted_boxes`` and ``previous_ref_bbox`` use normalized cxcywh.
        The four returned standard deviations correspond to normalized
        ``(x1, y1, x2, y2)`` coordinates.
        """
        if self.bbox_uncertainty_head is None:
            raise RuntimeError("BBox uncertainty head is not enabled.")
        if decoder_payload is None:
            raise RuntimeError(
                "Decoder features required by the bbox uncertainty head "
                "were not collected."
            )

        decoder_features = decoder_payload["decoder_output"]
        previous_ref_bbox = decoder_payload["previous_ref_bbox"]
        if decoder_features.shape[:2] != predicted_boxes.shape[:2]:
            raise ValueError(
                "Uncertainty decoder features do not match predicted boxes."
            )
        if previous_ref_bbox.shape != predicted_boxes.shape:
            raise ValueError(
                "Previous references do not match predicted box shape."
            )

        probabilities = logits.sigmoid()
        confidence = probabilities.amax(dim=-1)
        area = (
            predicted_boxes[..., 2].clamp_min(self.eps)
            * predicted_boxes[..., 3].clamp_min(self.eps)
        )
        log_area = area.log()
        abs_refinement = (predicted_boxes - previous_ref_bbox).abs()

        if self.bbox_uncertainty_use_tu:
            if bbox_tu is None:
                raise RuntimeError(
                    "This bbox uncertainty input mode requires bbox TU values."
                )
            valid_mask = torch.isfinite(bbox_tu)
            # log1p reduces scale skew while preserving TU ordering.
            tu_feature = torch.log1p(
                torch.nan_to_num(
                    bbox_tu,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp_min(0)
            )
        else:
            valid_mask = torch.ones_like(confidence, dtype=torch.bool)
            tu_feature = torch.zeros_like(confidence)

        if self.bbox_uncertainty_input_mode == "tu_only":
            decoder_features = torch.zeros_like(decoder_features)
            confidence = torch.zeros_like(confidence)
            log_area = torch.zeros_like(log_area)
            abs_refinement = torch.zeros_like(abs_refinement)

        if self.bbox_uncertainty_detach_inputs:
            decoder_features = decoder_features.detach()
            confidence = confidence.detach()
            log_area = log_area.detach()
            abs_refinement = abs_refinement.detach()
            tu_feature = tu_feature.detach()

        pred_std = self.bbox_uncertainty_head(
            decoder_features,
            tu_feature.to(decoder_features.dtype),
            confidence.to(decoder_features.dtype),
            log_area.to(decoder_features.dtype),
            abs_refinement.to(decoder_features.dtype),
        )
        pred_std = torch.where(
            valid_mask.unsqueeze(-1),
            pred_std,
            torch.full_like(pred_std, float("nan")),
        )
        return pred_std, valid_mask

    def freeze_detector_for_bbox_uncertainty(self):
        """Freeze RT-DETR and leave only the uncertainty head trainable."""
        if self.bbox_uncertainty_head is None:
            raise RuntimeError(
                "Enable bbox_uncertainty_enabled before freezing the model."
            )

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.bbox_uncertainty_head.parameters():
            parameter.requires_grad_(True)

        # Deterministic detector features; only the small head stays in train
        # mode. Call this after any outer trainer calls model.train().
        self.eval()
        self.bbox_uncertainty_head.train()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0) # type: ignore
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for _cls, _reg in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(_cls.bias, bias)
            init.constant_(_reg.layers[-1].weight, 0)
            init.constant_(_reg.layers[-1].bias, 0)
        
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m in self.input_proj:
            init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), 
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask


    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           denoising_logits=None,
                           denoising_bbox_unact=None):

        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask

        # memory = torch.where(valid_mask, memory, 0)
        # TODO fix type error for onnx export 
        memory = valid_mask.to(memory.dtype) * memory  

        output_memory :torch.Tensor = self.enc_output(memory)
        enc_outputs_logits :torch.Tensor = self.enc_score_head(output_memory)
        enc_outputs_coord_unact :torch.Tensor = self.enc_bbox_head(output_memory) + anchors

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_bbox_unact = \
            self._select_topk(output_memory, enc_outputs_logits, enc_outputs_coord_unact, self.num_queries)
            
        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        # if self.num_select_queries != self.num_queries:            
        #     raise NotImplementedError('')

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()
            
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
        
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)
        
        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_coords_unact: torch.Tensor, topk: int):
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
        
        topk_ind: torch.Tensor

        topk_coords = outputs_coords_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_coords_unact.shape[-1]))
        
        topk_logits = outputs_logits.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]))
        
        topk_memory = memory.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_coords


    def forward(self, feats, targets=None, collect_tu=False):
        need_bbox_tu = (
            (self.tu_enabled and not self.training)
            or (
                self.bbox_uncertainty_enabled
                and self.bbox_uncertainty_use_tu
            )
        )
        need_decoder_payload = (
            collect_tu
            or need_bbox_tu
            or self.bbox_uncertainty_enabled
        )

        memory, spatial_shapes = self._get_encoder_input(feats)

        if self.training and self.num_denoising > 0:
            (
                denoising_logits,
                denoising_bbox_unact,
                attn_mask,
                dn_meta,
            ) = get_contrastive_denoising_training_group(
                targets,
                self.num_classes,
                self.num_queries,
                self.denoising_class_embed,
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
            )
        else:
            denoising_logits = None
            denoising_bbox_unact = None
            attn_mask = None
            dn_meta = None

        (
            init_ref_contents,
            init_ref_points_unact,
            enc_topk_bboxes_list,
            enc_topk_logits_list,
        ) = self._get_decoder_input(
            memory,
            spatial_shapes,
            denoising_logits,
            denoising_bbox_unact,
        )

        out_bboxes, out_logits, tu_payload = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            collect_tu=need_decoder_payload,
        )

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(
                out_bboxes,
                dn_meta["dn_num_split"],
                dim=2,
            )
            dn_out_logits, out_logits = torch.split(
                out_logits,
                dn_meta["dn_num_split"],
                dim=2,
            )
            if tu_payload is not None:
                # The decoder payload contains denoising queries first. The
                # uncertainty head is trained only on the regular queries,
                # whose Hungarian assignments are produced by the criterion.
                for key in (
                    "bbox_last_input",
                    "decoder_output",
                    "previous_ref_bbox",
                ):
                    _, tu_payload[key] = torch.split(
                        tu_payload[key],
                        dn_meta["dn_num_split"],
                        dim=1,
                    )

        out = {
            "pred_logits": out_logits[-1],
            "pred_boxes": out_bboxes[-1],
        }

        bbox_tu = None
        if need_bbox_tu:
            if tu_payload is None:
                raise RuntimeError(
                    "TU activation payload was not collected."
                )
            tu_topk = (
                self.bbox_uncertainty_train_tu_topk
                if (
                    self.bbox_uncertainty_head is not None
                    and self.bbox_uncertainty_head.training
                )
                else self.bbox_tu_topk
            )
            bbox_tu = self._calculate_bbox_tu_batch(
                out["pred_logits"],
                out["pred_boxes"],
                tu_payload["bbox_last_input"],
                topk=tu_topk,
            )
            out["bbox_tu"] = bbox_tu

        if self.bbox_uncertainty_enabled:
            pred_bbox_std, uncertainty_valid = (
                self._predict_bbox_uncertainty(
                    out["pred_logits"],
                    out["pred_boxes"],
                    tu_payload,
                    bbox_tu=bbox_tu,
                )
            )
            out["pred_bbox_std"] = pred_bbox_std
            out["pred_bbox_log_std"] = pred_bbox_std.log()
            out["bbox_uncertainty_valid"] = uncertainty_valid

        if collect_tu and not self.training:
            out["tu_activations"] = {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in tu_payload.items()
            }

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                out_logits[:-1],
                out_bboxes[:-1],
            )
            out["enc_aux_outputs"] = self._set_aux_loss(
                enc_topk_logits_list,
                enc_topk_bboxes_list,
            )
            out["enc_meta"] = {
                "class_agnostic": self.query_select_method == "agnostic"
            }

            if dn_meta is not None:
                out["dn_aux_outputs"] = self._set_aux_loss(
                    dn_out_logits,
                    dn_out_bboxes,
                )
                out["dn_meta"] = dn_meta

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]

