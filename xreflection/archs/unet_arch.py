from xreflection.utils.registry import ARCH_REGISTRY
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
U-Net 구현 코드
https://dacon.io/codeshare/4245

Bottleneck channel 1024 -> 512로 수정
Bottleneck에 Attention (Residual) 추가

Unpool (Upsample): ConvTranspose2d -> interpolation='nearest'로 수정
- 체커보드 artifact 감소 목적으로, Stable Diffusion 및 최신 U-Net 논문에서 권장하는 방법
"""

class SelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.size()
        
        # Reshape: B x C x H x W -> B x (H*W) x C
        x_flat = x.view(B, C, H * W).transpose(1, 2)
        
        # Self-attention (query = key = value)
        attn_out, _ = self.mha(x_flat, x_flat, x_flat, need_weights=False)
        
        # Reshape back: B x (H*W) x C -> B x C x H x W
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        
        out = self.gamma * attn_out + x
        
        return out
    
@ARCH_REGISTRY.register()
class UNet(nn.Module):
    def __init__(self, ch=3):
        super(UNet, self).__init__()

        # Convolution + BatchNormalization + Relu 정의하기
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True): 
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # 수축 경로(Contracting path)
        self.enc1_1 = CBR2d(in_channels=ch, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # Bottleneck with Attention
        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.attention = SelfAttention(in_channels=512)  # Attention 추가

        # 확장 경로(Expansive path)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=ch, kernel_size=1, stride=1, padding=0, bias=True)
    
    # forward 함수 정의하기
    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_1 = self.attention(enc4_1)  # Self-Attention 적용

        dec4_1 = self.dec4_1(enc4_1)

        unpool3 = F.interpolate(dec4_1, size=enc3_2.shape[2:], mode="nearest")
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = F.interpolate(dec3_1, size=enc2_2.shape[2:], mode="nearest")
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = F.interpolate(dec2_1, size=enc1_2.shape[2:], mode="nearest")
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x
