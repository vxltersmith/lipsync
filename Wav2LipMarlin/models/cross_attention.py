import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, heads=8):
        super(CrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(dim_q, heads)
        self.proj_kv = nn.Linear(dim_kv, dim_q)
        self.norm = nn.LayerNorm(dim_q)
    
    def forward(self, q, kv):
        # q: [B, C, H, W], kv: [B, C_audio, 1, 1]
        B, C_q, H, W = q.shape
        q = q.view(B, C_q, H * W).permute(2, 0, 1)  # [N, B, C_q]
        
        # Reshape kv for cross-attention (audio embedding is 1x1)
        kv = kv.view(B, kv.shape[1]).unsqueeze(0)  # [1, B, C_kv] if kv is global
        
        q = self.attn(q, kv, kv)[0]
        q = q.permute(1, 2, 0).view(B, C_q, H, W)  # Reshape back to original
        return q