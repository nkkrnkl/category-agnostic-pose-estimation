# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# ------------------------------------------------------------------------------------

# For CAPE project, we only use poly2seq mode with roomformer_v2
from .roomformer_v2 import build as build_v2


def build_model(args, train=True, tokenizer=None):
    # CAPE project always uses poly2seq mode
    return build_v2(args, train, tokenizer=tokenizer)

