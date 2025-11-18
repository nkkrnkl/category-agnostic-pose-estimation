"""
Stub for polygon operations (legacy).

These functions were used for floor plan polygon data but are not needed
for CAPE keypoint pose estimation. Provided as stubs to avoid import errors.
"""


def get_all_order_corners(*args, **kwargs):
    """
    Stub function for polygon corner ordering.

    This function was used for floor plan polygon data.
    For CAPE keypoint pose estimation, this is not used.
    """
    raise NotImplementedError(
        "get_all_order_corners is not implemented for CAPE keypoint pose estimation. "
        "This function was only used for legacy polygon floor plan data."
    )
