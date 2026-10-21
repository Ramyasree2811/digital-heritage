import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowAttention(nn.Module):
    """
    Standard and Shifted Window multi-head self-attention (W-MSA / SW-MSA).
    """
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Query, Key, Value projection
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, mask=None):
        """
        x: input features with shape (num_windows * B, N, C)
        mask: (0/-inf) mask for shifted window attention, shape (num_windows, N, N)
        """
        b_win, n, c = x.shape
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(b_win, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # shape: (B_win, heads, N, head_dim)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            nw = mask.shape[0]
            # Reshape attention map to separate batch and windows
            attn = attn.reshape(b_win // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape(b_win, self.num_heads, n, n)
            
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(b_win, n, c)
        x = self.proj(x)
        return x

class SwinTransformerBlock(nn.Module):
    """
    A Swin Transformer Block containing W-MSA / SW-MSA and MLP.
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        if min(self.input_resolution) <= self.window_size:
            # If window size is larger than image resolution, don't shift
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads)
        
        self.norm2 = nn.LayerNorm(dim)
        # MLP / FFN
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def create_mask(self, h, w, device):
        # Calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, h, w, 1), device=device)  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h_s in h_slices:
            for w_s in w_slices:
                img_mask[:, h_s, w_s, :] = cnt
                cnt += 1

        # Partition mask into windows
        mask_windows = self.window_partition(img_mask, self.window_size)  # nW, window_size*window_size, 1
        mask_windows = mask_windows.squeeze(-1)  # nW, window_size*window_size
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def window_partition(self, x, window_size):
        b, h, w, c = x.shape
        x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, c)
        return windows

    def window_reverse(self, windows, window_size, h, w):
        b = int(windows.shape[0] / (h * w / window_size / window_size))
        x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
        return x

    def forward(self, x):
        h, w = self.input_resolution
        b, l, c = x.shape
        assert l == h * w, "Input feature size does not match resolution"

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # Shifted Window Mask setup
        if self.shift_size > 0:
            # Shift features
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # Partition windows
            x_windows = self.window_partition(shifted_x, self.window_size)
            # Create mask
            attn_mask = self.create_mask(h, w, x.device)
        else:
            shifted_x = x
            x_windows = self.window_partition(shifted_x, self.window_size)
            attn_mask = None

        # Apply attention inside windows
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows back
        if self.shift_size > 0:
            shifted_x = self.window_reverse(attn_windows, self.window_size, h, w)
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = self.window_reverse(attn_windows, self.window_size, h, w)

        x = x.view(b, h * w, c)
        x = shortcut + x

        # MLP branch
        x = x + self.mlp(self.norm2(x))
        return x

class SwinIR(nn.Module):
    """
    Lightweight SwinIR network for general image restoration and super-resolution.
    Uses Swin Transformer Blocks in a shallow channel configuration.
    """
    def __init__(self, in_channels=3, out_channels=3, img_size=256, channels=32, num_heads=4):
        super().__init__()
        self.img_size = img_size
        self.channels = channels
        
        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        
        # Deep feature extraction: Swin Transformer blocks
        # Stage 1: Window size 8, no shift
        self.swin1 = SwinTransformerBlock(channels, (img_size, img_size), num_heads, window_size=8, shift_size=0)
        # Stage 2: Window size 8, shift size 4
        self.swin2 = SwinTransformerBlock(channels, (img_size, img_size), num_heads, window_size=8, shift_size=4)
        
        # Final conv reconstruction
        self.conv_last = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Resize input to match target size if different
        b, c, h, w = x.shape
        resize_needed = (h != self.img_size or w != self.img_size)
        if resize_needed:
            orig_size = (h, w)
            x_res = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        else:
            x_res = x
            
        inp = x_res
        
        # Conv feature extraction
        feat = self.conv_first(x_res)
        
        # Reshape to token format (B, L, C)
        b_s, c_s, h_s, w_s = feat.shape
        feat_tokens = feat.flatten(2).transpose(1, 2)
        
        # Apply Swin blocks
        feat_tokens = self.swin1(feat_tokens)
        feat_tokens = self.swin2(feat_tokens)
        
        # Reshape back to spatial (B, C, H, W)
        feat_out = feat_tokens.transpose(1, 2).reshape(b_s, c_s, h_s, w_s)
        
        # Predict output residual
        out = self.conv_last(feat_out) + inp
        out = torch.clamp(out, 0.0, 1.0)
        
        # Interpolate back to original resolution if it was resized
        if resize_needed:
            out = F.interpolate(out, size=orig_size, mode='bilinear', align_corners=False)
            
        return out

if __name__ == "__main__":
    model = SwinIR()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print("SwinIR test successful! Output shape:", out.shape)
