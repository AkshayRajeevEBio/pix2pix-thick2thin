import torch
import torch.nn as nn
import torchvision.models as models

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_index: int = 9):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:layer_index]
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Normalize using ImageNet stats
        self.register_buffer('mean', torch.tensor(
            [0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('std', torch.tensor(
            [0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1) / 2  # De-normalize from [-1,1] to [0,1]
        x = (x - self.mean) / self.std
        return self.vgg(x)
