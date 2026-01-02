import torch
import torch.nn as nn

from .layers.pfn_encoder import RecencyBias, PFNEncoderLayerRecency, ZeroRecencyBias
from .layers.pfn_backbone import PFNBackbone
from .layers.patch_embedder import InputEncoder
from .layers.mask_builder import MaskBuilder

class LinearPFN(nn.Module):
    def __init__(self, L=20, H=10, d=256, L_blk=6, n_heads=8, d_ff=1024, dropout=0.1, recency_init=1e-2):
        super().__init__()
        self.L, self.H, self.d = L, H, d
        self.enc = InputEncoder(L, H, d)
        self.recency = RecencyBias(n_heads, init_alpha=recency_init, learnable=True)
        # self.recency = ZeroRecencyBias(n_heads)
        self.backbone = PFNBackbone(d, n_heads, d_ff, dropout, self.recency, L_blk)
        self.masker = MaskBuilder()
        self.head = nn.Linear(d, H)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, ctx_x, ctx_z, qry_x, t_ctx, t_qry):
        ctx_e, qry_e = self.enc(ctx_x, ctx_z, qry_x)              # [B,C,d],[B,Q,d]
        Z = torch.cat([ctx_e, qry_e], dim=1)                      # [B,S,d]
        t_all, base_mask = self.masker(t_ctx, t_qry)              # [B,S],[B,S,S]
        Z = self.backbone(Z, t_all, base_mask)                    # [B,S,d]
        C = ctx_x.shape[1]
        U = Z[:, C:, :]                                           # [B,Q,d]
        return self.head(U)                                       # [B,Q,H]