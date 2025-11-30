# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# ------------------------------------------------------------------------------------

# For CAPE project, we only use poly2seq mode with roomformer_v2
from .roomformer_v2 import build as build_v2


def build_model(args, train=True, tokenizer=None):
    if not args.poly2seq:
        return build_v2(args, train)
    # CAPE project always uses poly2seq mode
    # ========================================================================
    # CRITICAL FIX: Always set cape_mode=False for the base RoomFormerV2 model
    # ========================================================================
    # When cape_mode=True, RoomFormerV2 creates its own SupportPoseEncoder and
    # re-encodes support graphs internally. This DISCARDS the output from
    # CAPEModel's GeometricSupportEncoder (which has GCN, spatial PE, etc.).
    #
    # By setting cape_mode=False:
    # 1. RoomFormerV2 does NOT create SupportPoseEncoder (no duplicate encoder)
    # 2. CAPEModel injects GeometricSupportEncoder output into decoder
    # 3. The proper geometric encoder is actually used!
    #
    # The CAPEModel wrapper handles all CAPE-specific support conditioning.
    # ========================================================================
    return build_v2(args, train, tokenizer=tokenizer, cape_mode=False)

