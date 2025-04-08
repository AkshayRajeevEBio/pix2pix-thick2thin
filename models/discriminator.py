import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        def block(in_c, out_c, stride=2, normalize=True):
            layers = [nn.utils.spectral_norm(
                nn.Conv2d(in_c, out_c, 4, stride, 1, bias=False)
            )]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.block1 = block(in_channels * 2, 64, normalize=False)
        self.block2 = block(64, 128)
        self.block3 = block(128, 256)
        self.block4 = block(256, 512, stride=1)
        self.final = nn.Conv2d(512, 1, 4, stride=1, padding=1)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.final(x)

    def forward_with_features(self, x, y):
        features = []
        x = torch.cat([x, y], dim=1)

        x = self.block1(x)
        features.append(x)

        x = self.block2(x)
        features.append(x)

        x = self.block3(x)
        features.append(x)

        x = self.block4(x)
        features.append(x)

        out = self.final(x)
        return out, features
