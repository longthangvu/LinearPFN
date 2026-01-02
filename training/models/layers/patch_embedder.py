import torch
import torch.nn as nn

class InputEncoder(nn.Module):
    def __init__(self, L, H, d):
        super().__init__()
        self.ctx = nn.Linear(L + H, d)
        self.qry = nn.Linear(L + H, d)
        nn.init.xavier_uniform_(self.ctx.weight); nn.init.zeros_(self.ctx.bias)
        nn.init.xavier_uniform_(self.qry.weight); nn.init.zeros_(self.qry.bias)

    def forward(self, ctx_x, ctx_z, qry_x):  # [B,C,L],[B,C,H],[B,Q,L]
        B, Q, H = qry_x.shape[0], qry_x.shape[1], ctx_z.shape[-1]
        zeros = torch.zeros(B, Q, H, device=qry_x.device, dtype=qry_x.dtype)
        ctx_in = torch.cat([ctx_x, ctx_z], dim=-1)        # [B,C,L+H]
        qry_in = torch.cat([qry_x, zeros], dim=-1)        # [B,Q,L+H]
        return self.ctx(ctx_in), self.qry(qry_in)         # [B,C,d],[B,Q,d]