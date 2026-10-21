import torch
import torch.nn as nn
import torch.nn.functional as F

class MDTA(nn.Module):
    """
    Multi-Dconv Head-Transposed Attention (MDTA) from Restormer.
    Applies self-attention across channels instead of spatial dimensions.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dw = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for channel-wise attention
        # Shape: (B, heads, C/heads, H*W)
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        # L2 normalize Q and K
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Attention: Q @ K^T -> Transposed attention map of shape (B, heads, C/heads, C/heads)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)

        # Apply attention to V: attn @ V
        out = (attn @ v)
        
        # Reshape back to spatial dimensions
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)
        return out

class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network (GDFN) from Restormer.
    Uses depthwise convolutions and a gating mechanism.
    """
    def __init__(self, channels, expansion_factor=2.0):
        super().__init__()
        expanded_channels = int(channels * expansion_factor)
        
        self.project_in = nn.Conv2d(channels, expanded_channels * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(expanded_channels * 2, expanded_channels * 2, kernel_size=3, padding=1, groups=expanded_channels * 2, bias=False)
        
        self.project_out = nn.Conv2d(expanded_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        # Gating mechanism: GELU(x1) * x2
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class RestormerBlock(nn.Module):
    """Restormer Transformer Block: MDTA -> GDFN with LayerNorm and Skip Connections"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels)

    def forward(self, x):
        # Apply LayerNorm along the channel dimension
        # PyTorch LayerNorm applies to last dims, so permute to (B, H, W, C)
        b, c, h, w = x.shape
        
        # Self-Attention Branch
        x_norm = x.permute(0, 2, 3, 1) # (B, H, W, C)
        x_norm = self.norm1(x_norm).permute(0, 3, 1, 2) # back to (B, C, H, W)
        x = x + self.attn(x_norm)
        
        # Feed-Forward Branch
        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.norm2(x_norm).permute(0, 3, 1, 2)
        x = x + self.ffn(x_norm)
        
        return x

class Restormer(nn.Module):
    """
    Lightweight Restormer model for general image restoration.
    Uses 3 stages of Restormer blocks in a U-shape encoder-decoder.
    """
    def __init__(self, in_channels=3, out_channels=3, num_blocks=[1, 1, 1], channels=32):
        super().__init__()
        
        self.intro = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        
        # Stage 1: Encoder
        self.encoder1 = nn.Sequential(*[RestormerBlock(channels) for _ in range(num_blocks[0])])
        self.down1 = nn.Conv2d(channels, channels * 2, kernel_size=4, stride=2, padding=1)  # 256 -> 128
        
        # Stage 2: Encoder
        self.encoder2 = nn.Sequential(*[RestormerBlock(channels * 2) for _ in range(num_blocks[1])])
        self.down2 = nn.Conv2d(channels * 2, channels * 4, kernel_size=4, stride=2, padding=1)  # 128 -> 64
        
        # Stage 3: Bottleneck
        self.bottleneck = nn.Sequential(*[RestormerBlock(channels * 4) for _ in range(num_blocks[2])])
        
        # Stage 2: Decoder
        self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=2, stride=2)  # 64 -> 128
        self.decoder2 = nn.Sequential(*[RestormerBlock(channels * 2) for _ in range(num_blocks[1])])
        
        # Stage 1: Decoder
        self.up1 = nn.ConvTranspose2d(channels * 2, channels, kernel_size=2, stride=2)  # 128 -> 256
        self.decoder1 = nn.Sequential(*[RestormerBlock(channels) for _ in range(num_blocks[0])])
        
        self.ending = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Residual learning: predict a residual image
        inp = x
        x = self.intro(x)
        
        # Encoder Stage 1
        e1 = self.encoder1(x)
        
        # Encoder Stage 2
        e2 = self.encoder2(self.down1(e1))
        
        # Bottleneck
        b = self.bottleneck(self.down2(e2))
        
        # Decoder Stage 2
        d2 = self.up2(b)
        # Skip connection 2 (residual sum)
        d2 = self.decoder2(d2 + e2)
        
        # Decoder Stage 1
        d1 = self.up1(d2)
        # Skip connection 1 (residual sum)
        d1 = self.decoder1(d1 + e1)
        
        out = self.ending(d1) + inp
        return torch.clamp(out, 0.0, 1.0)

if __name__ == "__main__":
    model = Restormer()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print("Restormer test successful! Output shape:", out.shape)
