"""U-Net with a nerve-present classification head (segmentation_models_pytorch)."""
import segmentation_models_pytorch as smp
import torch

# intensity normalisation applied to uint8 pixels; fixed so the browser app can reproduce it
PIX_MEAN, PIX_STD = 99.0, 55.7


def build_model(encoder='mobilenet_v2', weights='imagenet', decoder=(128, 64, 32, 16, 8)):
    return smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=1,
                    decoder_channels=decoder, aux_params=dict(classes=1, dropout=0.2))


class Exportable(torch.nn.Module):
    """uint8-range float input (N,1,H,W) -> (mask probability, presence probability)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        m, c = self.net((x - PIX_MEAN) / PIX_STD)
        return torch.sigmoid(m), torch.sigmoid(c)
