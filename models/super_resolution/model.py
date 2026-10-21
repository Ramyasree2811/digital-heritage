import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)

class SuperResolutionNet(nn.Module):
    """
    Lightweight Super-Resolution network representing the Real-ESRGAN enhancement module.
    Upscales the restored image by a factor of 2x using PixelShuffle.
    """
    def __init__(self, in_channels=3, out_channels=3, num_features=32, num_blocks=4):
        super().__init__()
        
        self.conv_first = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )
        
        # Upscaling block: PixelShuffle(2)
        # Upscaling requires out_channels * scale^2 features. For scale=2, it requires num_features -> out_channels * 4 features.
        self.conv_up = nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        self.conv_last = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.relu(self.conv_first(x))
        feat = self.res_blocks(feat)
        
        # Upscale 2x
        feat_up = self.pixel_shuffle(self.conv_up(feat))
        out = self.conv_last(self.relu(feat_up))
        
        return self.sigmoid(out)

if __name__ == "__main__":
    model = SuperResolutionNet()
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    print("SuperResolutionNet test successful! Output shape:", out.shape)
