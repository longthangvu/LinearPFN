import torch
import torch.nn as nn

class MaskBuilder(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.jit.export
    def struct_mask(self, C: int, Q: int, device: torch.device):
        S = C + Q
        m = torch.zeros(S, S, dtype=torch.bool, device=device)
        m[:C, C:] = True
        m[C:, C:] = True
        return m  # [S,S], True=disallow

    def forward(self, t_ctx: torch.Tensor, t_qry: torch.Tensor):
        # t_ctx: [B,C] long; t_qry: [B,Q] long
        B, C = t_ctx.shape
        Q = t_qry.shape[1]
        device = t_ctx.device
        t_all = torch.cat([t_ctx, t_qry], dim=1)                  # [B,S]
        struct = self.struct_mask(C, Q, device)[None, ...].expand(B, -1, -1)
        nofuture = (t_all.unsqueeze(2) < t_all.unsqueeze(1))      # [B,S,S]
        return t_all, (struct | nofuture)                         # [B,S], [B,S,S]

