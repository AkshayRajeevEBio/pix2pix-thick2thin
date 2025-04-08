import torch
import torch.nn.functional as F

def compute_centroid(mask: torch.Tensor, target_color: torch.Tensor) -> tuple:
    """
    Find the centroid of all pixels matching target_color in a (3, H, W) tensor.
    If no match is found, returns center of the image.

    Args:
        mask (Tensor): RGB image as tensor of shape (3, H, W)
        target_color (Tensor): RGB values as tensor (3,)

    Returns:
        tuple: (x, y) centroid
    """
    match = (mask == target_color.view(3, 1, 1))
    mask_binary = match.all(dim=0).float()

    indices = torch.nonzero(mask_binary, as_tuple=True)
    if indices[0].numel() == 0:
        centroid_y = mask.shape[1] // 2
        centroid_x = mask.shape[2] // 2
    else:
        centroid_y = torch.mean(indices[0].float()).item()
        centroid_x = torch.mean(indices[1].float()).item()

    return centroid_x, centroid_y


class CenterAndPad:
    def __init__(self, lumen_index, size=(75, 75), constant_values=0):
        """
        Args:
            lumen_index (tuple): RGB color to center around, e.g., (255, 0, 0)
            size (tuple): Desired (H, W) size after padding
            constant_values (int or tuple): Padding color/value
        """
        self.lumen_index = torch.tensor(lumen_index, dtype=torch.uint8)
        self.size = size
        self.constant_values = constant_values

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image (Tensor): shape (3, H, W), RGB tensor
        Returns:
            Tensor: padded image centered around lumen_index
        """
        c, h, w = image.shape
        desired_h, desired_w = self.size

        # 1. Compute centroid of target color
        cx, cy = compute_centroid(image, self.lumen_index)

        cx = int(cx)
        cy = int(cy)

        # 2. Compute necessary padding
        pad_left = max(0, (desired_w // 2) - cx)
        pad_right = max(0, desired_w - (w + pad_left))
        pad_top = max(0, (desired_h // 2) - cy)
        pad_bottom = max(0, desired_h - (h + pad_top))

        # 3. Apply padding using F.pad (pads in reverse order: [W_left, W_right, H_top, H_bottom])
        padding = [pad_left, pad_right, pad_top, pad_bottom]
        image_padded = F.pad(
            image, padding, mode='constant', value=self.constant_values
        )

        return image_padded

