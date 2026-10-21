import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """
    Lightweight U-Net architecture for general image restoration.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)  # 128 (up) + 128 (skip) = 256
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)   # 64 (up) + 64 (skip) = 128
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)     # 32 (up) + 32 (skip) = 64
        
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()  # Map output to [-1, 1] if input is normalized to [-1, 1], or use Sigmoid if [0, 1]
        
    def forward(self, x):
        # input is typically in [0, 1] or [-1, 1]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder
        x = self.up1(x4)
        # Skip connection 1
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        # Skip connection 2
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        # Skip connection 3
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        
        logits = self.outc(x)
        # Residual learning: add input back to help model learn differences (optional, but let's keep it direct)
        return torch.sigmoid(logits)  # Map to [0, 1]
        
if __name__ == "__main__":
    model = UNet()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print("UNet test successful! Output shape:", out.shape)
