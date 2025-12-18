from xreflection.utils.registry import ARCH_REGISTRY
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Guided U-Net:
    - Conv 대신 NAFBlock 사용
    - Bottleneck에 GuideFusionBlock 추가하여 VLM 임베딩 주입 (Cross-Attention)
    - Upsample은 ConvTranspose2d 사용
"""

# ==============================================================================
# 1. Architecture Components (SimpleGate, NAFBlock, GuideFusionBlock)
# ==============================================================================

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    """
    State-of-the-Art Image Restoration Block
    복잡한 활성화 함수 없이 Gating과 Channel Attention만으로 고성능을 냄
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1)
        )

        self.sg = SimpleGate()
        
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)

        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)
        
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        input = x
        x = x.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C) for LayerNorm
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2) # Back to (B, C, H, W)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        
        y = input + x * self.beta

        x = y.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma

class GuideFusionBlock(nn.Module):
    """
    Bottleneck에서 Guide Embedding을 Cross-Attention으로 주입하는 블록
    Query: Image Features
    Key, Value: Guide Embedding
    """
    def __init__(self, dim, guide_dim=2048, num_heads=8):
        super().__init__()
        self.guide_proj = nn.Linear(guide_dim, dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
    def forward(self, x, guide_emb):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2) # (B, H*W, C)
        
        # Guide Embedding 준비 (B, 2048) -> (B, 1, C)
        context = self.guide_proj(guide_emb).unsqueeze(1) 
        
        # Cross Attention (Q=Image, K=Guide, V=Guide)
        attn_out, _ = self.cross_attn(query=x_flat, key=context, value=context)
        
        # Residual Connection & Reshape
        out = x_flat + attn_out
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out

# ==============================================================================
# 2. Main Network Architecture (nn.Module)
# ==============================================================================

@ARCH_REGISTRY.register()
class GuidedUNet(nn.Module):
    def __init__(self, in_channels=3, base_c=32, guide_dim=2048):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, base_c, 3, 1, 1), NAFBlock(base_c))
        self.down1 = nn.Conv2d(base_c, base_c*2, 2, 2)
        
        self.enc2 = NAFBlock(base_c*2)
        self.down2 = nn.Conv2d(base_c*2, base_c*4, 2, 2)
        
        self.enc3 = NAFBlock(base_c*4)
        self.down3 = nn.Conv2d(base_c*4, base_c*8, 2, 2)
        
        # Bottleneck with Guide Fusion
        self.bottleneck = NAFBlock(base_c*8)
        self.fusion = GuideFusionBlock(dim=base_c*8, guide_dim=guide_dim)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_c*8, base_c*4, 2, 2)
        self.reduce3 = nn.Conv2d(base_c*8, base_c*4, 1) 
        self.dec3 = NAFBlock(base_c*4)

        self.up2 = nn.ConvTranspose2d(base_c*4, base_c*2, 2, 2)
        self.reduce2 = nn.Conv2d(base_c*4, base_c*2, 1)
        self.dec2 = NAFBlock(base_c*2)

        self.up1 = nn.ConvTranspose2d(base_c*2, base_c, 2, 2)
        self.reduce1 = nn.Conv2d(base_c*2, base_c, 1)
        self.dec1 = NAFBlock(base_c)
        
        self.final = nn.Conv2d(base_c, 3, 3, 1, 1)

    def forward(self, x, guide_emb):
        # Encoding
        e1 = self.enc1(x)         
        e2 = self.enc2(self.down1(e1)) 
        e3 = self.enc3(self.down2(e2)) 
        b = self.down3(e3)             
        
        # Bottleneck & Fusion
        b = self.bottleneck(b)
        b = self.fusion(b, guide_emb) # VLM 정보 주입
        
        # Decoding
        d3 = self.up3(b)               
        d3 = torch.cat([d3, e3], dim=1) # Skip Connection
        d3 = self.dec3(self.reduce3(d3))
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(self.reduce2(d2))
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(self.reduce1(d1))
        
        # [중요] 이 Output은 'Reflection(반사)' 성분을 예측합니다.
        out = self.final(d1)
        return out
    
