from xreflection.utils.registry import ARCH_REGISTRY
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    """
    Sinusoidal time embedding.
    t: (B,) in [0,1]
    returns: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class PatchEmbed(nn.Module):
    """
    Conv patch embedding: (B,C,H,W) -> (B, N, D)
    """
    def __init__(self, in_chans, embed_dim, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                 # (B, D, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2) # (B, N, D)
        return x


class PatchUnembedDynamic(nn.Module):
    """
    Linear projection back to patches then fold dynamically:
      (B, N, D) + (H, W) -> (B, C, H', W') where H',W' are padded sizes divisible by patch_size.
    """
    def __init__(self, embed_dim, out_chans, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.proj = nn.Linear(embed_dim, out_chans * patch_size * patch_size)

    def forward(self, x, H, W):
        B, N, D = x.shape
        ps = self.patch_size
        gh, gw = H // ps, W // ps
        assert N == gh * gw, f"Token count mismatch: N={N}, expected {gh*gw} (gh={gh}, gw={gw})"

        x = self.proj(x)  # (B, N, C*ps*ps)
        x = x.view(B, gh, gw, self.out_chans, ps, ps)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # (B, C, gh, ps, gw, ps)
        x = x.view(B, self.out_chans, gh * ps, gw * ps)
        return x


class DiTBlock(nn.Module):
    """
    Minimal DiT-style block with AdaLN-Zero modulation from conditioning vector c.
    Uses:
      x = x + tanh(gate_attn) * Attn(LN(x) * (1+scale1) + shift1)
      x = x + tanh(gate_mlp)  * MLP(LN(x) * (1+scale2) + shift2)
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

        # produces: shift1, scale1, gate1, shift2, scale2, gate2
        self.ada = nn.Linear(embed_dim, 6 * embed_dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, c):
        """
        x: (B, N, D)
        c: (B, D)
        """
        B, N, D = x.shape
        params = self.ada(c).view(B, 6, D)
        shift1, scale1, gate1, shift2, scale2, gate2 = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4], params[:, 5]

        # Attention
        h = self.norm1(x)
        h = h * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + torch.tanh(gate1).unsqueeze(1) * h

        # MLP
        h2 = self.norm2(x)
        h2 = h2 * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        h2 = self.mlp(h2)
        x = x + torch.tanh(gate2).unsqueeze(1) * h2
        return x


@ARCH_REGISTRY.register()
class DualHeadDiTRectifiedFlow(nn.Module):
    """
    Dynamic-resolution DiT for Rectified Flow (dual head).

    Input:
      x_t: (B, 6, H, W)   state = concat(x_T, x_R)
      t:   (B,)           scalar in [0,1]
      I:   (B, 3, H, W)   condition image (observation)

    Output:
      v_T: (B, 3, H, W)
      v_R: (B, 3, H, W)

    Notes:
      - Supports arbitrary H,W via padding to multiples of patch_size.
      - Uses 2D pos_embed learned on base img_size grid and interpolated to (H/ps, W/ps).
    """
    def __init__(self, img_size=224, patch_size=16, in_chans_state=6, in_chans_cond=3, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_chans_state = in_chans_state
        self.in_chans_cond = in_chans_cond

        in_chans = in_chans_state + in_chans_cond  # 9 channels total by concat conditioning
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size)

        # 2D positional embedding learned on base grid (img_size/patch_size)
        base_grid = img_size // patch_size
        self.pos_embed_2d = nn.Parameter(torch.zeros(1, embed_dim, base_grid, base_grid))
        nn.init.trunc_normal_(self.pos_embed_2d, std=0.02)

        # time embedding -> conditioning vector c
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

        # dynamic unpatch
        self.unpatch = PatchUnembedDynamic(embed_dim, out_chans=in_chans_state, patch_size=patch_size)

    def _pad_to_patch_multiple(self, x, ps):
        B, C, H, W = x.shape
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, pad_h, pad_w

    def forward(self, x_t, t, I):
        """
        x_t: (B,6,H,W)
        t:   (B,)
        I:   (B,3,H,W)
        """
        B, C, H, W = x_t.shape
        assert C == self.in_chans_state, f"x_t channels mismatch: got {C}, expected {self.in_chans_state}"
        assert I.shape[0] == B and I.shape[1] == self.in_chans_cond, "I shape mismatch"
        assert I.shape[2] == H and I.shape[3] == W, "x_t and I spatial size must match"

        ps = self.patch_size

        # pad both to multiples of patch_size (so patch embedding is well-defined)
        x_t_pad, pad_h, pad_w = self._pad_to_patch_multiple(x_t, ps)
        I_pad, _, _ = self._pad_to_patch_multiple(I, ps)

        H2, W2 = x_t_pad.shape[2], x_t_pad.shape[3]
        gh, gw = H2 // ps, W2 // ps

        # concat state + condition
        x_in = torch.cat([x_t_pad, I_pad], dim=1)  # (B, 9, H2, W2)

        # tokens
        tok = self.patch_embed(x_in)  # (B, N, D) where N = gh*gw

        # interpolate pos embed to current grid and add
        pos = F.interpolate(self.pos_embed_2d, size=(gh, gw), mode="bilinear", align_corners=False)  # (1,D,gh,gw)
        pos = pos.flatten(2).transpose(1, 2)  # (1, N, D)
        tok = tok + pos

        # conditioning vector from time
        if t.dim() != 1:
            t = t.view(-1)
        te = timestep_embedding(t, self.embed_dim)  # (B, D)
        c = self.time_mlp(te)  # (B, D)

        for blk in self.blocks:
            tok = blk(tok, c)

        tok = self.final_norm(tok)

        v = self.unpatch(tok, H2, W2)  # (B, 6, H2, W2)

        # crop back to original size if padded
        if pad_h or pad_w:
            v = v[:, :, :H, :W]

        v_T, v_R = v[:, :3], v[:, 3:]
        return v_T, v_R
