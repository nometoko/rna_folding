

import torch
import torch.nn as nn

class dMAELoss(nn.Module):
    """
    Mean Absolute Error on pairwise distances.
    """
    def __init__(self, epsilon: float = 1e-4, Z: float = 10.0):
        super().__init__()
        self.epsilon = epsilon
        self.Z = Z

    def calculate_distance_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # X, Y: (L, 3)
        diff = X.unsqueeze(1) - Y.unsqueeze(0)     # (L, L, 3)
        return torch.sqrt((diff ** 2).sum(-1) + self.epsilon)  # (L, L)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        pred, gt: (B, L, 3)
        Returns scalar loss.
        """
        batch_size = pred.size(0)
        losses = []
        eye = None
        for b in range(batch_size):
            pred_b = pred[b]  # (L, 3)
            gt_b = gt[b]      # (L, 3)
            pred_dm = self.calculate_distance_matrix(pred_b, pred_b)
            gt_dm   = self.calculate_distance_matrix(gt_b, gt_b)
            # mask out diagonal and NaNs
            if eye is None:
                L = pred_dm.size(0)
                eye = torch.eye(L, dtype=torch.bool, device=pred_dm.device)
            mask = ~torch.isnan(gt_dm) & ~eye
            diff = torch.abs(pred_dm[mask] - gt_dm[mask])
            losses.append(diff.mean() / self.Z)
        return torch.stack(losses).mean()
