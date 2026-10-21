import torch
import torch.nn as nn

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class Pix2PixGenerator(nn.Module):
    """
    U-Net Generator for Pix2Pix.
    Downsamples the input image and then upsamples it with skip connections.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder
        self.down1 = UNetDown(in_channels, 32, normalize=False)  # 256 -> 128
        self.down2 = UNetDown(32, 64)                            # 128 -> 64
        self.down3 = UNetDown(64, 128)                           # 64 -> 32
        self.down4 = UNetDown(128, 256)                          # 32 -> 16
        self.down5 = UNetDown(256, 256)                          # 16 -> 8
        self.down6 = UNetDown(256, 256, normalize=False)         # 8 -> 4
        
        # Decoder
        self.up1 = UNetUp(256, 256, dropout=0.5)                 # 4 -> 8
        self.up2 = UNetUp(512, 256, dropout=0.5)                 # 8 -> 16
        self.up3 = UNetUp(512, 128)                              # 16 -> 32
        self.up4 = UNetUp(256, 64)                               # 32 -> 64
        self.up5 = UNetUp(128, 32)                               # 64 -> 128
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),       # 128 -> 256
            nn.Sigmoid()                                         # Map to [0, 1]
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        
        return self.final(u5)

class Pix2PixDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for Pix2Pix.
    Determines whether overlapping patches of size 70x70 are real or fake.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # Concatenate input image and restored image along channel dimension
            *discriminator_block(in_channels + out_channels, 32, normalization=False), # 256 -> 128
            *discriminator_block(32, 64),                                             # 128 -> 64
            *discriminator_block(64, 128),                                            # 64 -> 32
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 1, 4, padding=1, bias=False)                               # Outputs patch prediction map
        )

    def forward(self, img_a, img_b):
        # Concatenate images by channels
        img_input = torch.cat((img_a, img_b), 1)
        return self.model(img_input)

if __name__ == "__main__":
    generator = Pix2PixGenerator()
    discriminator = Pix2PixDiscriminator()
    x = torch.randn(1, 3, 256, 256)
    fake_y = generator(x)
    print("Pix2Pix Generator output shape:", fake_y.shape)
    pred = discriminator(x, fake_y)
    print("Pix2Pix Discriminator output shape:", pred.shape)
