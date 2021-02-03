import torch
import torch.nn as nn
import torch.fft

class LLayerfft(nn.Module):
    
    def __init__(self, mask_mode, mask_type, sparsity, h = 32, w = 32):
        super(LLayerfft, self).__init__()
        #img_size = (img.size()).detach().numpy()
        #width = img_size[2]
        #height = img_size[3]
        self.sparsity = sparsity
        self.weights = torch.rand(1, 1, h ,w)
        self.mask_mode = mask_mode
        self.mask_type = mask_type
        if self.mask_mode == "varied": 
            self.mask = nn.Parameter(torch.rand(1, 1, h, w))
            if self.mask_type == "1d":
                for i in range(h):
                    self.mask[:, :, i, :]= torch.mean(self.mask[:, :, i, :])
        else: #mask_mode == "fixed"
            if self.mask_type == "2drandom":
                self.mask = torch.rand(1, 1, h ,w)
            elif self.mask_type == "1drandom":
                self.mask = torch.rand(1, 1, h ,w)
                for i in range(h):
                    self.mask[:, :, i, :]= torch.mean(self.mask[:, :, i, :])
            else:
                self.mask = torch.rand(1, 1, h ,w)
    
    def cast(self, x):
        return(x.type(torch.cuda.FloatTensor) if torch.cuda.is_available() else x.type(torch.FloatTensor))

    
    def forward(self, img, epoch, mode):
        if mode == "train" and self.mask_mode == "varied":
            sparse_mask = self.mask - torch.quantile(self.mask, 1 - self.sparsity)
            binary_mask = torch.sigmoid((2**epoch)* sparse_mask)
            self.weights = self.cast(binary_mask)   
        else: #mode == "test"
            sparse_mask = self.mask - torch.quantile(self.mask, 1 - self.sparsity)
            binary_mask = torch.where(sparse_mask > 0, 1, 0)
            self.weights = self.cast(binary_mask)     
        fftn = self.cast(torch.fft.ifftn(self.weights * torch.fft.fftn(img)))
        return fftn 
    
