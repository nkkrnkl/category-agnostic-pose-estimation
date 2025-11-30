"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from util.poly_ops import get_all_order_corners
class HungarianMatcher(nn.Module):
    """
    Computes an assignment between targets and predictions.
    """
    def __init__(self,
                 cost_class: float = 1,
                 cost_coords: float = 1):
        """
        Initialize matcher.
        
        Args:
            cost_class: Relative weight of classification error
            cost_coords: Relative weight of coordinate error
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_coords = cost_coords
        assert cost_class != 0 or cost_coords != 0, "all costs cant be 0"
    def calculate_angles(self, polygon):
        vect1 = polygon.roll(1, 0)-polygon
        vect2 = polygon.roll(-1, 0)-polygon
        cos_sim = ((vect1 * vect2).sum(1)+1e-9)/(torch.norm(vect1, p=2, dim=1)*torch.norm(vect2, p=2, dim=1)+1e-9)
        return cos_sim
    def calculate_src_angles(self, polygon):
        vect1 = polygon.roll(1, 1)-polygon
        vect2 = polygon.roll(-1, 1)-polygon
        cos_sim = ((vect1 * vect2).sum(-1)+1e-9)/(torch.norm(vect1, p=2, dim=-1)*torch.norm(vect2, p=2, dim=-1)+1e-9)
        return cos_sim
    def forward(self, outputs, targets):
        """
        Perform matching.
        
        Args:
            outputs: Dict with pred_logits and pred_coords
            targets: List of target dicts with labels and coords
        
        Returns:
            List of tuples (index_i, index_j) for each batch element
        """
        with torch.no_grad():
            bs, num_polys = outputs["pred_logits"].shape[:2]
            src_prob = outputs["pred_logits"].flatten(0,1).sigmoid()
            src_polys = outputs["pred_coords"].flatten(0, 1).flatten(1, 2)
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_polys = torch.cat([v["coords"] for v in targets])
            tgt_len = torch.cat([v["lengths"] for v in targets])
            cost_class = torch.cdist(src_prob, tgt_ids, p=1)
            cost_coords = torch.zeros([src_polys.shape[0], tgt_polys.shape[0]], device=src_polys.device)
            for i in range(tgt_polys.shape[0]):
                tgt_polys_single = tgt_polys[i, :tgt_len[i]]
                all_polys = get_all_order_corners(tgt_polys_single)
                cost_coords[:, i] = torch.cdist(src_polys[:, :tgt_len[i]], all_polys , p=1).min(axis=-1)[0]
            C = self.cost_coords * cost_coords + self.cost_class * cost_class
            C = C.view(bs, num_polys, -1).cpu()
            sizes = [len(v["coords"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
def build_matcher(args):
    """
    Build matcher from arguments.
    
    Args:
        args: Arguments with set_cost_class and set_cost_coords
    
    Returns:
        HungarianMatcher instance
    """
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_coords=args.set_cost_coords)