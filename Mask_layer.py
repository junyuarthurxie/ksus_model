import torch
import torch.nn as nn
import torch.fft
import masks
import numpy as np

def FFTshift(img):
    h = list(img.shape)[2]
    w = list(img.shape)[3]
    img_buffer = torch.clone(img[0, 0, :, :])
    img_new = torch.clone(img[0, 0, :, :])
    img_buffer[0 : np.int(h / 2), 0 : np.int(w / 2)] = img_new[np.int(h / 2) : h,  np.int(w / 2) : w]
    img_buffer[np.int(h / 2) : h,  np.int(w / 2) : w] = img_new[0 : np.int(h / 2), 0 : np.int(w / 2)]
    img_buffer[0 : np.int(h / 2), np.int(w / 2) : w] = img_new[np.int(h / 2) : h,  0 : np.int(w / 2)]
    img_buffer[np.int(h / 2) : h,  0 : np.int(w / 2)] = img_new[0 : np.int(h / 2), np.int(w / 2) : w]
    img[0, 0, :, :] = img_buffer
    return img

def cast(x):
        return(x.type(torch.cuda.FloatTensor) if torch.cuda.is_available() else x.type(torch.FloatTensor))
    
def mask_to_tensor(mask):
    return cast(torch.from_numpy(np.fft.fftshift(np.resize(mask, (1, 1, mask.shape[0], mask.shape[1])))))

class mask_layer(nn.Module):
    
    def __init__(self, mask_mode, mask_type, sparsity, h, w):
        super(mask_layer, self).__init__()
        #img_size = (img.size()).detach().numpy()
        #width = img_size[2]
        #height = img_size[3]
        self.sparsity = sparsity
        self.weights = torch.rand(1, 1, h ,w)
        self.sweights = torch.rand(1, 1, h ,w)
        self.mask_mode = mask_mode
        self.mask_type = mask_type
        self.sub_ks = torch.rand(1, 1, h ,w)
        self.ks = torch.rand(1, 1, h ,w)
        #self.thres = self.cast(torch.rand(1, 1, h ,w))
        if self.mask_mode == "varied" or self.mask_mode == "varied2": 
            rand_number = torch.rand(1, 1, h, w)
            self.mask = nn.Parameter(rand_number)
        else: #mask_mode == "fixed"
            
            if self.mask_type == "2drandom":
                self.weights = mask_to_tensor(masks.random2d(h, w, sparsity)) 
            elif self.mask_type == "1drandom":
                self.weights = mask_to_tensor(masks.random1d(h, w, sparsity))
            elif self.mask_type == "1drandomc":
                self.weights = mask_to_tensor(masks.random1dc(h, w, sparsity))
            elif self.mask_type == "2drandomc":
                self.weights = mask_to_tensor(masks.random2dc(h, w, sparsity))
            elif self.mask_type == "1duniformc":
                self.weights = mask_to_tensor(masks.uniform1dc(h, w, sparsity))
            elif self.mask_type == "1duniform":
                self.weights = mask_to_tensor(masks.uniform1d(h, w, sparsity))
            elif self.mask_type == "2dgauss":
                self.weights = mask_to_tensor(masks.gauss2d(h, w, sparsity))
            elif self.mask_type == "2dgaussc":
                self.weights = mask_to_tensor(masks.gauss2dc(h, w, sparsity))
            else:
                print("No mask detected")
    
    
    def forward(self, img, slope, mode):
        if mode == "train" and self.mask_mode == "varied":
            rand_number = torch.rand(1, 1, img.shape[2], img.shape[3])
            sigmoid_mask = torch.sigmoid(slope * self.mask)
            sparse_mask = sigmoid_mask * self.sparsity /torch.mean(sigmoid_mask)
            binary_mask = torch.sigmoid(slope * (sparse_mask - cast(rand_number)))
            self.weights = cast(binary_mask)   
            #self.sweights = FFTshift(self.weights)
        elif mode == "test" and self.mask_mode == "varied":
            rand_number = torch.rand(1, 1, img.shape[2], img.shape[3])
            sigmoid_mask = torch.sigmoid(slope * self.mask)
            sparse_mask = sigmoid_mask * self.sparsity /torch.mean(sigmoid_mask)
            binary_mask = torch.where(sparse_mask > cast(rand_number), 1, 0)
            self.weights = cast(binary_mask)
            #self.sweights = FFTshift(self.weights)
        elif mode == "train" and self.mask_mode == "varied2":
            sparse_mask = self.mask - (torch.quantile(self.mask, 1 - self.sparsity) )
            binary_mask = torch.sigmoid(slope * sparse_mask)
            self.weights = cast(binary_mask)
            #self.sweights = FFTshift(self.weights)
        elif mode == "test" and self.mask_mode == "varied2": 
            sparse_mask = self.mask - (torch.quantile(self.mask, 1 - self.sparsity)) 
            binary_mask = torch.where(sparse_mask > 0, 1, 0)
            self.weights = cast(binary_mask)
            #self.sweights = FFTshift(self.weights)
        else:
            self.weights = self.weights
            
        self.ks = torch.clone(FFTshift(torch.fft.fftn(img)))
        self.sub_ks = FFTshift(torch.clone(self.weights)) * self.ks
        fftn = cast(torch.fft.ifftn(self.weights * torch.fft.fftn(img)))
        return fftn 
    

