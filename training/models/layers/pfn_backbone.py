import torch.nn as nn
from .pfn_encoder import PFNEncoderLayerRecency

class PFNBackbone(nn.Module):
    def __init__(self, d, n_heads, d_ff, dropout, recency, L_blk):
        super().__init__()
        self.blocks = nn.ModuleList([
            PFNEncoderLayerRecency(d, n_heads, d_ff, dropout, recency)
            for _ in range(L_blk)
        ])

    def forward(self, Z, t_all, base_mask):  # Z:[B,S,d]
        for blk in self.blocks:
            Z = blk(Z, t_all, base_mask)
        return Z