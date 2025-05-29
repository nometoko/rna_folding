import torch
import torch.nn as nn

class dMAELoss(nn.Module):
    """
    Distance-based Mean Absolute Error (dMAE)
    ─────────────────────────────────────────
    • pred, gt            : (B, L, 3)  あるいは (L, 3)
      - バッチ axis B は省略可
      - L がバッチごとに異なっても OK（短い方に合わせて自動で切りそろえます）
    • epsilon             : 距離のゼロ割り回避用
    • Z                   : 正規化定数
    """
    def __init__(self, epsilon: float = 1e-4, Z: float = 10.0):
        super().__init__()
        self.epsilon = epsilon
        self.Z = Z

    # ------------------------------------------------------------
    # private helper
    # ------------------------------------------------------------
    def _pairwise_dist(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (..., L, 3)
        → (..., L, L)  - 3D ユークリッド距離
        """
        diff = x.unsqueeze(-2) - x.unsqueeze(-3)         # (..., L, L, 3)
        return torch.sqrt((diff ** 2).sum(-1) + self.epsilon)

    # ------------------------------------------------------------
    # forward
    # ------------------------------------------------------------
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        pred, gt: (B, L, 3) or (L, 3)
        returns : scalar loss
        """
        # バッチが無い場合はダミー次元を追加
        if pred.dim() == 2:
            pred, gt = pred.unsqueeze(0), gt.unsqueeze(0)

        losses = []
        for p, g in zip(pred, gt):
            # ── 1. 長さをそろえる（短い方で切る） ─────────────────────────
            L = min(p.size(0), g.size(0))
            p, g = p[:L], g[:L]

            # ── 2. NaN を含む点を除外（両方で有効な index のみ残す） ───────
            valid = ~(torch.isnan(p).any(-1) | torch.isnan(g).any(-1))
            if valid.sum() < 2:                      # 距離計算に十分でない
                continue
            p, g = p[valid], g[valid]

            # ── 3. 距離行列 → 対角を除外 → MAE ───────────────────────────
            pdm = self._pairwise_dist(p)
            gdm = self._pairwise_dist(g)

            eye = torch.eye(pdm.size(0), dtype=torch.bool, device=pdm.device)
            mae = torch.abs(pdm[~eye] - gdm[~eye]).mean() / self.Z
            losses.append(mae)

        # バッチ平均（valid が無ければ 0.0）
        return torch.stack(losses).mean() if losses else pred.new_tensor(0.0)
