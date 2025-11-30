import os
import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np
from util.misc import NestedTensor, nested_tensor_from_tensor_list, interpolate, inverse_sigmoid
from datasets.token_types import TokenType
from .backbone import build_backbone
from .matcher import build_matcher
from .losses import custom_L1_loss, MaskRasterizationLoss
try:
    from .label_smoothing_loss import label_smoothed_nll_loss
except ImportError:
    def label_smoothed_nll_loss(logits, target, epsilon=0.0, ignore_index=-100):
        """
        Compute cross-entropy loss with optional label smoothing.
        
        Args:
            logits: Raw logits, shape (N, num_classes)
            target: Target class indices, shape (N,)
            epsilon: Label smoothing factor
            ignore_index: Index to ignore in loss computation
        
        Returns:
            Loss value
        """
        if logits.dim() == 1:
            raise ValueError(f"logits should be 2D, got shape {logits.shape}")
        if target.dim() > 1:
            target = target.view(-1)
        if logits.size(0) != target.size(0):
            raise ValueError(f"logits and target batch size mismatch: {logits.size(0)} vs {target.size(0)}")
        if epsilon > 0:
            log_probs = F.log_softmax(logits, dim=-1)
            n_classes = logits.size(-1)
            target_one_hot = torch.zeros_like(log_probs)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
            target_one_hot = target_one_hot * (1 - epsilon) + epsilon / n_classes
            loss = -(target_one_hot * log_probs).sum(dim=-1)
            return loss.mean()
        else:
            return F.cross_entropy(logits, target, reduction='mean')
from .deformable_transformer_v2 import build_deforamble_transformer
import copy
def _get_clones(module, N):
    """
    Clone a module N times.
    
    Args:
        module: Module to clone
        N: Number of clones
    
    Returns:
        ModuleList of cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
class SupportPoseEncoder(nn.Module):
    """
    Encodes the Support Pose Graph.
    """
    def __init__(self, hidden_dim=256, nhead=8, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(2, hidden_dim)
        self.kps_pos_embed = nn.Embedding(100, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=1024, 
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, support_coords, support_mask=None):
        """
        Forward pass.
        
        Args:
            support_coords: Support coordinates, shape (B, N_kps, 2)
            support_mask: Optional mask, shape (B, N_kps)
        
        Returns:
            support_features: Support features, shape (B, N_kps, Hidden_Dim)
        """
        x = self.input_proj(support_coords)
        B, N, _ = x.shape
        pos = self.kps_pos_embed(torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1))
        x = x + pos
        if support_mask is not None:
            src_key_padding_mask = ~support_mask
        else:
            src_key_padding_mask = None
        support_features = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return support_features
class RoomFormerV2(nn.Module):
    """
    RoomFormer module for floorplan reconstruction.
    """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_polys, num_feature_levels,
                 aux_loss=True, with_poly_refine=False, masked_attn=False, semantic_classes=-1, seq_len=1024, tokenizer=None,
                 use_anchor=False, patch_size=1, freeze_anchor=False, inject_cls_embed=False,
                 cape_mode=False
                 ):
        """
        Initialize the model.
        
        Args:
            backbone: Backbone module
            transformer: Transformer module
            num_classes: Number of object classes
            num_queries: Number of object queries
            num_polys: Maximal number of possible polygons
            aux_loss: Whether to use auxiliary decoding losses
            with_poly_refine: Whether to use iterative polygon refinement
            masked_attn: Whether to use masked attention
            semantic_classes: Number of semantic classes
            seq_len: Sequence length
            tokenizer: Tokenizer instance
            use_anchor: Whether to use anchor
            patch_size: Patch size
            freeze_anchor: Whether to freeze anchor
            inject_cls_embed: Whether to inject CLS embedding
            cape_mode: Whether to use CAPE mode
        """
        super().__init__()
        self.num_queries = num_queries
        self.num_polys = num_polys
        assert  num_queries % num_polys == 0
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.cape_mode = cape_mode
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.coords_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.num_feature_levels = num_feature_levels
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.inject_cls_embed = inject_cls_embed
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size, padding=0),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                if patch_size == 1:
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
                else:
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=2*patch_size, stride=2*patch_size, padding=0),
                        nn.GroupNorm(32, hidden_dim),
                    ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_poly_refine = with_poly_refine
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.coords_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.coords_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        num_pred = transformer.decoder.num_layers
        if with_poly_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.coords_embed = _get_clones(self.coords_embed, num_pred)
            nn.init.constant_(self.coords_embed[0].layers[-1].bias.data[2:], -2.0)
        else:
            nn.init.constant_(self.coords_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.coords_embed = nn.ModuleList([self.coords_embed for _ in range(num_pred)])
        if use_anchor or with_poly_refine:
            self.query_embed = nn.Embedding(seq_len, 2)
            self.query_embed.weight.requires_grad = not freeze_anchor
        else:
            self.query_embed = None
        self.transformer.decoder.coords_embed = self.coords_embed
        self.transformer.decoder.class_embed = self.class_embed
        self.room_class_embed = None
        if semantic_classes > 0:
            self.room_class_embed = nn.Linear(hidden_dim, semantic_classes)
            if self.inject_cls_embed:
                self.transformer.decoder.room_class_embed = self.room_class_embed
        self.register_buffer('attention_mask', self._create_causal_attention_mask(seq_len))
        if self.cape_mode:
            print("Initializing CAPE Support Pose Encoder...")
            self.support_encoder = SupportPoseEncoder(hidden_dim=hidden_dim, num_layers=3)
            self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)
    def _create_causal_attention_mask(self, seq_len):
        """
        Creates a causal attention mask for a sequence of length `seq_len`.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        causal_mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, 0.0)
        return causal_mask
    def forward(self, samples: NestedTensor, seq_kwargs=None, support_graphs=None, support_mask=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [batch_size x C x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_coords": The normalized corner coordinates for all queries, represented as
                               (x, y). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        bs = samples.tensors.shape[0]
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            src = self.input_proj[l](src)
            srcs.append(src)
            if self.patch_size != 1:
                mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos[l] = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        query_embeds = None if self.query_embed is None else self.query_embed.weight
        tgt_embeds = None
        support_embeddings = None
        if self.cape_mode and support_graphs is not None:
            support_embeddings = self.support_encoder(support_graphs, support_mask=support_mask)
        hs, init_reference, inter_references, inter_classes = self.transformer(
            srcs, masks, pos, query_embeds, tgt_embeds, self.attention_mask, seq_kwargs,
            support_features=support_embeddings
        )
        num_layer = hs.shape[0]
        outputs_class = inter_classes
        outputs_coord = inter_references
        out = {'pred_logits': outputs_class[-1], 'pred_coords': outputs_coord[-1]}
        if self.room_class_embed is not None:
            outputs_room_class = self.room_class_embed(hs[-1])
            out = {'pred_logits': outputs_class[-1], 'pred_coords': outputs_coord[-1], 'pred_room_logits': outputs_room_class}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out
    def _prepare_sequences(self, b):
        prev_output_token_11 = [[self.tokenizer.bos] for _ in range(b)]
        prev_output_token_12 = [[self.tokenizer.bos] for _ in range(b)]
        prev_output_token_21 = [[self.tokenizer.bos] for _ in range(b)]
        prev_output_token_22 = [[self.tokenizer.bos] for _ in range(b)]
        delta_x1 = [[0] for _ in range(b)]
        delta_y1 = [[0] for _ in range(b)]
        delta_x2 = [[1] for _ in range(b)]
        delta_y2 = [[1] for _ in range(b)]
        gen_out = [[] for _ in range(b)]
        if self.inject_cls_embed:
            input_polygon_labels = [[self.semantic_classes-1] for _ in range(b)]
        else:
            input_polygon_labels = [[-1] for _ in range(b)]
        return (
            prev_output_token_11, prev_output_token_12, prev_output_token_21, prev_output_token_22,
            delta_x1, delta_x2, delta_y1, delta_y2,
            gen_out, input_polygon_labels
        )
    def forward_inference(self, samples: NestedTensor, use_cache=True, support_graphs=None, support_mask=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [batch_size x C x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_coords": The normalized corner coordinates for all queries, represented as
                               (x, y). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        bs = samples.tensors.shape[0]
        support_embeddings = None
        if self.cape_mode and support_graphs is not None:
            support_embeddings = self.support_encoder(support_graphs, support_mask=support_mask)
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            src = self.input_proj[l](src)
            srcs.append(src)
            if self.patch_size != 1:
                mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos[l] = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        if use_cache:
            max_src_len = sum([x.size(2) * x.size(3) for x in srcs])
            self._setup_caches(bs, max_src_len)
        (prev_output_token_11, prev_output_token_12, prev_output_token_21, prev_output_token_22,
            delta_x1, delta_x2, delta_y1, delta_y2,
            gen_out, input_polygon_labels) = self._prepare_sequences(bs)
        query_embeds = None if self.query_embed is None else self.query_embed.weight
        tgt_embeds = None
        enc_cache = None
        device = samples.tensors.device
        num_bins = self.tokenizer.num_bins
        min_len = 6
        max_len = self.tokenizer.seq_len
        unfinish_flag = np.ones(bs)
        i = 0
        output_hs_list = []
        output_cls_list = []
        output_reg_list = []
        DEBUG_KEYPOINT_BUG = os.environ.get('DEBUG_KEYPOINT_BUG', '0') == '1'
        if DEBUG_KEYPOINT_BUG and bs > 0:
            print(f"\n[DEBUG_KEYPOINT_BUG] Starting autoregressive generation:")
            print(f"  Batch size: {bs}")
            print(f"  Max sequence length: {max_len}")
            print(f"  Min sequence length: {min_len}")
        while i < max_len and unfinish_flag.any():
            prev_output_tokens_11_tensor = torch.tensor(np.array(prev_output_token_11)[:, i:i+1]).to(device).long()
            prev_output_tokens_12_tensor = torch.tensor(np.array(prev_output_token_12)[:, i:i+1]).to(device).long()
            prev_output_tokens_21_tensor = torch.tensor(np.array(prev_output_token_21)[:, i:i+1]).to(device).long()
            prev_output_tokens_22_tensor = torch.tensor(np.array(prev_output_token_22)[:, i:i+1]).to(device).long()
            delta_x1_tensor = torch.tensor(np.array(delta_x1)[:, i:i+1], dtype=torch.float32).to(device)
            delta_x2_tensor = torch.tensor(np.array(delta_x2)[:, i:i+1], dtype=torch.float32).to(device)
            delta_y1_tensor = torch.tensor(np.array(delta_y1)[:, i:i+1], dtype=torch.float32).to(device)
            delta_y2_tensor = torch.tensor(np.array(delta_y2)[:, i:i+1], dtype=torch.float32).to(device)
            input_polygon_labels_tensor = torch.tensor(np.array(input_polygon_labels)[:, i:i+1], dtype=torch.long).to(device)
            seq_kwargs = {
                'seq11': prev_output_tokens_11_tensor,
                'seq12': prev_output_tokens_12_tensor,
                'seq21': prev_output_tokens_21_tensor,
                'seq22': prev_output_tokens_22_tensor,
                'delta_x1': delta_x1_tensor,
                'delta_x2': delta_x2_tensor,
                'delta_y1': delta_y1_tensor,
                'delta_y2': delta_y2_tensor,
                'input_polygon_labels': input_polygon_labels_tensor
            }
            if not use_cache:
                hs, _, reg_output, cls_output = self.transformer(srcs, masks, pos, query_embeds, tgt_embeds, None, 
                                                                                    seq_kwargs, force_simple_returns=True, return_enc_cache=use_cache, 
                                                                                    enc_cache=None, decode_token_pos=None,
                                                                                    support_features=support_embeddings)
                output_hs_list.append(hs[:, i:i+1])
                output_cls_list.append(cls_output)
                output_reg_list.append(reg_output)
            else:
                decode_token_pos = torch.tensor([i], device=device, dtype=torch.long)
                hs, _, reg_output, cls_output, enc_cache = self.transformer(srcs, masks, pos, query_embeds, tgt_embeds, None, 
                                                                                    seq_kwargs, force_simple_returns=True, return_enc_cache=use_cache, 
                                                                                    enc_cache=enc_cache, decode_token_pos=decode_token_pos,
                                                                                    support_features=support_embeddings)
                output_hs_list.append(hs)
                output_cls_list.append(cls_output)
                output_reg_list.append(reg_output)
            cls_type = torch.argmax(cls_output, 2)
            if DEBUG_KEYPOINT_BUG and i < 10 and bs > 0:
                cls_0 = cls_type[0, 0].item()
                token_name = {0: 'COORD', 1: 'SEP', 2: 'CLS', 3: 'EOS', 4: 'PAD'}.get(cls_0, f'UNKNOWN({cls_0})')
                print(f"  Step {i}: Predicted token type = {token_name}")
            for j in range(bs):
                if unfinish_flag[j] == 1:
                    cls_j = cls_type[j, 0].item()
                    if cls_j == TokenType.coord.value or (cls_j == TokenType.eos.value and i < min_len):
                        output_j_x, output_j_y = reg_output[j, 0].detach().cpu().numpy()
                        output_j_x = min(output_j_x, 1)
                        output_j_y = min(output_j_y, 1)
                        gen_out[j].append([output_j_x, output_j_y])
                        output_j_x = output_j_x * (num_bins - 1)
                        output_j_y = output_j_y * (num_bins - 1)
                        output_j_x_floor = math.floor(output_j_x)
                        output_j_y_floor = math.floor(output_j_y)
                        output_j_x_ceil = math.ceil(output_j_x)
                        output_j_y_ceil = math.ceil(output_j_y)
                        prev_output_token_11[j].append(output_j_x_floor * num_bins + output_j_y_floor)
                        prev_output_token_12[j].append(output_j_x_floor * num_bins + output_j_y_ceil)
                        prev_output_token_21[j].append(output_j_x_ceil * num_bins + output_j_y_floor)
                        prev_output_token_22[j].append(output_j_x_ceil * num_bins + output_j_y_ceil)
                        delta_x = output_j_x - output_j_x_floor
                        delta_y = output_j_y - output_j_y_floor
                    elif cls_j == TokenType.sep.value:
                        gen_out[j].append(2)
                        prev_output_token_11[j].append(self.tokenizer.sep)
                        prev_output_token_12[j].append(self.tokenizer.sep)
                        prev_output_token_21[j].append(self.tokenizer.sep)
                        prev_output_token_22[j].append(self.tokenizer.sep)
                        delta_x = 0
                        delta_y = 0
                    elif cls_j == TokenType.cls.value:
                        gen_out[j].append(-1)
                        prev_output_token_11[j].append(self.tokenizer.cls)
                        prev_output_token_12[j].append(self.tokenizer.cls)
                        prev_output_token_21[j].append(self.tokenizer.cls)
                        prev_output_token_22[j].append(self.tokenizer.cls)
                        delta_x = 0
                        delta_y = 0
                    else:
                        unfinish_flag[j] = 0
                        gen_out[j].append(-1)
                        prev_output_token_11[j].append(self.tokenizer.eos)
                        prev_output_token_12[j].append(self.tokenizer.eos)
                        prev_output_token_21[j].append(self.tokenizer.eos)
                        prev_output_token_22[j].append(self.tokenizer.eos)
                        delta_x = 0
                        delta_y = 0
                else:
                    gen_out[j].append(-1)
                    prev_output_token_11[j].append(self.tokenizer.pad)
                    prev_output_token_12[j].append(self.tokenizer.pad)
                    prev_output_token_21[j].append(self.tokenizer.pad)
                    prev_output_token_22[j].append(self.tokenizer.pad)
                    delta_x = 0
                    delta_y = 0
                delta_x1[j].append(delta_x)
                delta_y1[j].append(delta_y)
                delta_x2[j].append(1 - delta_x)
                delta_y2[j].append(1 - delta_y)
            i += 1
        if os.environ.get('DEBUG_KEYPOINT_COUNT', '0') == '1':
            print(f"[DIAG roomformer_v2] Generation finished at step {i}/{max_len}")
            print(f"  unfinish_flag: {unfinish_flag}")
            print(f"  gen_out[0] length: {len(gen_out[0])}")
            print(f"  output_cls_list length: {len(output_cls_list)}")
            print(f"  output_reg_list length: {len(output_reg_list)}")
        incomplete_generations = unfinish_flag.sum()
        if incomplete_generations > 0 and os.environ.get('WARN_INCOMPLETE_GENERATION', '1') == '1':
            import warnings
            warnings.warn(
                f"⚠️  {int(incomplete_generations)}/{bs} sequences reached max_len={max_len} "
                f"without predicting EOS. This suggests the model hasn't learned proper "
                f"stopping behavior. Consider retraining with EOS token included in loss."
            )
        if len(output_cls_list) > 0:
            all_cls_output = torch.cat(output_cls_list, dim=1)
            all_reg_output = torch.cat(output_reg_list, dim=1)
            if DEBUG_KEYPOINT_BUG and bs > 0:
                print(f"\n[DEBUG_KEYPOINT_BUG] Generation complete:")
                print(f"  Total iterations: {i}")
                print(f"  gen_out[0] length: {len(gen_out[0])}")
                print(f"  all_cls_output shape: {all_cls_output.shape}")
                print(f"  all_reg_output shape: {all_reg_output.shape}")
                print(f"  First sample finished: {unfinish_flag[0] == 0}")
        else:
            all_cls_output = None
            all_reg_output = None
            if DEBUG_KEYPOINT_BUG:
                print(f"\n[DEBUG_KEYPOINT_BUG] WARNING: No iterations! output_cls_list is empty")
        out = {'pred_logits': all_cls_output, 'pred_coords': all_reg_output, 'gen_out': gen_out}
        if self.room_class_embed is not None:
            hs = torch.cat(output_hs_list, dim=1)
            outputs_room_class = self.room_class_embed(hs)
            out = {'pred_logits': all_cls_output, 'pred_coords': all_reg_output, 
                   'pred_room_logits': outputs_room_class, 'gen_out': gen_out, 
                   'anchors': query_embeds.detach()}
        if out['pred_coords'] is not None and len(gen_out) > 0:
            actual_len = out['pred_coords'].shape[1]
            expected_len = len(gen_out[0])
            if actual_len != expected_len:
                raise RuntimeError(
                    f"CRITICAL BUG DETECTED: forward_inference output shape mismatch!\n"
                    f"  Generated {expected_len} tokens in gen_out\n"
                    f"  But pred_coords only has {actual_len} positions\n"
                    f"  This indicates the output accumulation is broken."
                )
        return out
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_coords': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
    def _setup_caches(self, max_bs, max_src_len):
        self.transformer._setup_caches(max_bs, self.seq_len, max_src_len,
                                      self.transformer.d_model,
                                      self.transformer.nhead,
                                      self.transformer.level_embed.dtype,
                                      device=self.transformer.level_embed.device)
class SemHead(nn.Module):
    """
    Semantic head for room classification.
    """
    def __init__(self, hidden_dim, num_classes):
        """
        Initialize semantic head.
        
        Args:
            hidden_dim: Hidden dimension
            num_classes: Number of classes
        """
        super().__init__()
        self.shared_layer = nn.Linear(hidden_dim, hidden_dim)
        self.room_embed = nn.Linear(hidden_dim, num_classes-2)
        self.num_classes = num_classes
        self.window_door_embed = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor
        """
        x = F.normalize(torch.relu(self.shared_layer(x)), p=2, dim=-1, eps=1e-12)
        room_out = self.room_embed(x)
        window_door_out = self.window_door_embed(x)
        out = torch.cat([room_out[:,:,:-1], window_door_out, room_out[:,:,-1:]], dim=-1)
        return out.contiguous()
class Raster2Seq(RoomFormerV2):
    """
    Raster2Seq module for floorplan reconstruction.
    """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_polys, num_feature_levels,
                 aux_loss=True, with_poly_refine=False, masked_attn=False, semantic_classes=-1, seq_len=1024, tokenizer=None,
                 use_anchor=False,
                 ):
        super().__init__(backbone, transformer, num_classes, num_queries, num_polys, num_feature_levels,
                         aux_loss=aux_loss, with_poly_refine=with_poly_refine, masked_attn=masked_attn,
                         semantic_classes=semantic_classes, seq_len=seq_len, tokenizer=tokenizer,
                         use_anchor=use_anchor)
        hidden_dim = transformer.d_model
        self.room_class_embed = None
        if semantic_classes > 0:
            self.room_class_embed = SemHead(hidden_dim, semantic_classes)
class SetCriterion(nn.Module):
    """
    Computes the loss for multiple polygons.
    """
    def __init__(self, num_classes, semantic_classes, matcher, weight_dict, losses, label_smoothing=0.,
                 per_token_sem_loss=False, ):
        """
        Initialize criterion.
        
        Args:
            num_classes: Number of classes for corner validity
            semantic_classes: Number of semantic classes
            matcher: Matching module
            weight_dict: Dictionary of loss weights
            losses: List of losses to apply
            label_smoothing: Label smoothing factor
            per_token_sem_loss: Whether to compute semantic loss per token
        """
        super().__init__()
        self.num_classes = num_classes
        self.semantic_classes = semantic_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.label_smoothing = label_smoothing
        self.per_token_sem_loss = per_token_sem_loss
        if 'loss_raster' in self.weight_dict:
            self.raster_loss = MaskRasterizationLoss(None)
    def _update_ce_coeff(self, loss_ce_coeff):
        self.weight_dict['loss_ce'] = loss_ce_coeff
    def loss_labels(self, outputs, targets, indices):
        """
        Classification loss (NLL).
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            indices: Matching indices
        
        Returns:
            Dictionary with loss_ce and optionally loss_ce_room
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        bs = src_logits.shape[0]
        target_classes = targets['token_labels'].to(src_logits.device)
        mask = (target_classes != -1).bool()
        loss_ce = label_smoothed_nll_loss(src_logits[mask], target_classes[mask],
                                          epsilon=self.label_smoothing)
        losses = {'loss_ce': loss_ce}
        if 'pred_room_logits' in outputs:
            room_src_logits = outputs['pred_room_logits']
            if not self.per_token_sem_loss:
                mask = (target_classes == 3)
                room_target_classes = targets['target_polygon_labels'].to(room_src_logits.device)
                if mask.sum() > 0:
                    loss_ce_room = label_smoothed_nll_loss(room_src_logits[mask], room_target_classes[room_target_classes != -1], epsilon=self.label_smoothing)
                else:
                    loss_ce_room = torch.tensor(0.0, device=src_logits.device)
            else:
                room_target_classes = targets['target_polygon_labels'].to(room_src_logits.device)
                loss_ce_room = label_smoothed_nll_loss(room_src_logits[room_target_classes != -1], room_target_classes[room_target_classes
                    != -1], epsilon=self.label_smoothing)
            losses = {'loss_ce': loss_ce, 'loss_ce_room': loss_ce_room}
        return losses
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        """
        Compute the cardinality error.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            indices: Matching indices
        
        Returns:
            Dictionary with cardinality_error
        """
        losses = {'cardinality_error': 0.}
        return losses
    def _extract_polygons(self, sequence, token_labels):
        B, N = token_labels.shape
        polygons = []
        for b in range(B):
            labels = token_labels[b]
            coords = sequence[b]
            sep_eos_mask = (labels == 1) | (labels == 2)
            split_indices = torch.nonzero(sep_eos_mask, as_tuple=False).squeeze(-1)
            if len(split_indices) == 0:
                corner_mask = (labels == 0)
                if corner_mask.any():
                    polygons.append(coords[corner_mask])
                continue
            device = labels.device
            starts = torch.cat([torch.tensor([0], device=device), split_indices[:-1] + 1])
            ends = split_indices
            for s, e in zip(starts, ends):
                if s < e:
                    segment_labels = labels[s:e]
                    segment_coords = coords[s:e]
                    corner_mask = (segment_labels == 0)
                    if corner_mask.any():
                        polygons.append(segment_coords[corner_mask])
        return polygons
    def loss_polys(self, outputs, targets, indices):
        """
        Compute losses related to polygons.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            indices: Matching indices
        
        Returns:
            Dictionary with loss_coords and optionally loss_raster
        """
        assert 'pred_coords' in outputs
        bs = outputs['pred_coords'].shape[0]
        src_poly = outputs['pred_coords']
        device = src_poly.device
        token_labels = targets['token_labels'].to(device)
        mask = (token_labels == 0).bool()
        target_polys = targets['target_seq'].to(device)
        loss_coords = F.l1_loss(src_poly[mask], target_polys[mask])
        losses = {}
        losses['loss_coords'] = loss_coords
        if self.weight_dict.get('loss_raster', 0) > 0:
            pred_poly_list = self._extract_polygons(src_poly, token_labels)
            target_poly_list = self._extract_polygons(target_polys, token_labels)
            loss_raster_mask = self.raster_loss(pred_poly_list, target_poly_list, [len(x) for x in target_poly_list],)
            losses['loss_raster'] = loss_raster_mask
        return losses
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'polys': self.loss_polys
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)
    def forward(self, outputs, targets):
        """
        Perform loss computation.
        
        Args:
            outputs: Dictionary of output tensors
            targets: List of target dictionaries
        
        Returns:
            Dictionary of losses
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        indices = None
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, **kwargs))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            indices = None
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, targets, indices)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses
class MLP(nn.Module):
    """
    Multi-layer perceptron (also called FFN).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of layers
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
def build(args, train=True, tokenizer=None, cape_mode=False):
    """
    Build model and criterion.
    
    Args:
        args: Arguments
        train: Whether in training mode
        tokenizer: Tokenizer instance
        cape_mode: Whether to use CAPE mode
    
    Returns:
        Model and optionally criterion
    """
    num_classes = 3 if not args.add_cls_token else 4
    if tokenizer is not None:
        pad_idx = tokenizer.pad
    else:
        pad_idx = 0
    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args, pad_idx=pad_idx)
    if getattr(args, 'model_version', 'v1') == 'v1':
        model = RoomFormerV2(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            num_polys=args.num_polys,
            num_feature_levels=args.num_feature_levels,
            aux_loss=args.aux_loss,
            with_poly_refine=args.with_poly_refine,
            masked_attn=args.masked_attn,
            semantic_classes=args.semantic_classes,
            seq_len=args.seq_len,
            tokenizer=tokenizer,
            use_anchor=args.use_anchor,
            patch_size=[1, 2][args.image_size == 512],
            freeze_anchor=getattr(args, 'freeze_anchor', False),
            inject_cls_embed=getattr(args, 'inject_cls_embed', False),
            cape_mode=cape_mode,
        )
    else:
        model = Raster2Seq(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            num_polys=args.num_polys,
            num_feature_levels=args.num_feature_levels,
            aux_loss=args.aux_loss,
            with_poly_refine=args.with_poly_refine,
            masked_attn=args.masked_attn,
            semantic_classes=args.semantic_classes,
            seq_len=args.seq_len,
            tokenizer=tokenizer,
            use_anchor=args.use_anchor,
        )
    if not train:
        return model
    device = torch.device(args.device)
    matcher = None
    weight_dict = {
                    'loss_ce': args.cls_loss_coef, 
                    'loss_ce_room': args.room_cls_loss_coef,
                    'loss_coords': args.coords_loss_coef,
                    }
    if args.raster_loss_coef > 0:
        weight_dict['loss_raster'] = args.raster_loss_coef
    weight_dict['loss_dir'] = 1
    enc_weight_dict = {}
    enc_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
    weight_dict.update(enc_weight_dict)
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ['labels', 'polys', 'cardinality']
    criterion = SetCriterion(num_classes, args.semantic_classes, matcher, weight_dict, losses,
                             label_smoothing=args.label_smoothing, 
                             per_token_sem_loss=args.per_token_sem_loss)
    criterion.to(device)
    return model, criterion