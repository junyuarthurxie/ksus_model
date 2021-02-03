import torch
import torch.nn as nn
import U_net
import Mask_layer


class ksus_model(nn.Module):
    
    def __init__(self, mask_mode = "fixed", mask_type = "2drandom", sparsity = 0.1, h = 32, w = 32):
        super(ksus_model, self).__init__()
        self.mask_layer = Mask_layer.mask_layer(mask_mode, mask_type, sparsity, h, w)
        self.Unet = U_net.UNet()

    def forward(self, x, slope = 1, mode = "test"):
        x1 = self.mask_layer(x, slope, mode)
        x2 = self.Unet(x1)
        return x2


