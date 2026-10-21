import torch
import torch.nn as nn

class DnCNN(nn.Module):
    """
    Lightweight DnCNN architecture for image denoising and artifact removal.
    Predicts the residual noise present in the image.
    """
    def __init__(self, in_channels=3, out_channels=3, num_layers=12, num_features=32):
        super().__init__()
        
        layers = []
        # First layer: Conv + ReLU
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # Intermediate layers: Conv + BN + ReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
            
        # Last layer: Conv
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=False))
        
        self.dncnn = nn.Sequential(*layers)
        
    def forward(self, x):
        # Residual learning: Predict the noise
        noise = self.dncnn(x)
        # Denoisied Image = Input - Predicted Noise
        out = x - noise
        return torch.clamp(out, 0.0, 1.0)

if __name__ == "__main__":
    model = DnCNN()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print("DnCNN test successful! Output shape:", out.shape)
