[x] Learnable PE 
[x] post-fusion vs pre-fusion: Change orders of attent layers in decoder: cross-attn -> attn
[x] Concat tgt and src for self-attention
[x] Pre dec PE
[x] Support EMA
[x] tune dropout (e.g. 0.1), weight decay (e.g. 0.01), clip_norm (e.g. 1.)
[x] use_anchor and query_pos_type='sine' (query_pos is not None)
[] use_anchor and query_pos_type='none' and prePosEmbed (query_pos is None)
[] use_anchor and disable adding query_pos for tgt, pre