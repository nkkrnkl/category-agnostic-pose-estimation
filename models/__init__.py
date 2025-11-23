# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# ------------------------------------------------------------------------------------

# For CAPE project, we only use poly2seq mode with roomformer_v2
from .roomformer_v2 import build as build_v2


def build_model(args, train=True, tokenizer=None):
    if not args.poly2seq:
        return build_v2(args, train)
    # CAPE project always uses poly2seq mode
    # Check if we are in CAPE mode (you can add this arg to main.py)
    is_cape = getattr(args, 'cape_mode', False)
    return build_v2(args, train, tokenizer=tokenizer, cape_mode=is_cape)

