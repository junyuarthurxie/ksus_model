import os, os.path
import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

import KSUS
import mnist_reader
from data_processing import get_data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
dir_path = os.path.dirname(os.path.realpath(__file__))




# Setting test data
offset = 48#000
test_size = 10#00

_ , data_test, data_demo = get_data(offset, test_size)

h, w = data_test.shape[1], data_test.shape[2]


class ksus_Dataset(Dataset):
  def __init__(self,datasetclean):
    self.clean=datasetclean
  
  def __len__(self):
    return len(self.clean)
  
  def __getitem__(self,idx):
    xClean=self.clean[idx]
    return xClean 
 
testset = ksus_Dataset(data_test)
testloader = DataLoader(testset, batch_size = 32, shuffle = True)

# Loading model
name = "fixed_1duniformc_1.pt"
mask_mode = "fixed"
mask_type = "1duniformc"
sparsity = 0.1
model = KSUS.ksus_model(mask_mode = mask_mode, mask_type = mask_type, sparsity = sparsity, h = h, w = w)
model.to(device)
model.load_state_dict(torch.load(dir_path + '/model/' + name))
model.eval()

criterion = nn.MSELoss()
test_loss = 0
for i, data in enumerate(testloader, 0):
    img = data
    img = img.view(img.shape[0], 1, img.shape[1], img.shape[2]).type(torch.FloatTensor)
    img = img.to(device)
    #-----------------Forward Pass----------------------
    output = model(img)#, 16, "train")
    loss = criterion(output, img)
    test_loss += loss.item()

# k_space & sub_sampled_kspace
#ks = model.mask_layer.ks.detach().cpu().numpy()[0, 0, :, :]
#plt.figure()
#plt.imshow(np.log(np.abs(ks)), cmap = "gray")
sub_ks = model.mask_layer.sub_ks.detach().cpu().numpy()[0, 0, :, :]
plt.figure()
plt.imshow(np.log(np.abs(sub_ks)), cmap = "gray")

# Forced_binaried_mask & distribution
binary_weights = np.fft.fftshift(np.copy(model.mask_layer.weights.detach().cpu().numpy()[0, 0, :, :]))
plt.figure()
plt.imshow(binary_weights, cmap = "gray", vmin = 0, vmax = 1)
plt.figure()
plt.hist(np.resize(binary_weights,(h * w)), bins = 100)

# Reconstructed_image 
plt.figure()
plt.imshow((output.detach().cpu().numpy())[0, 0, :, :], cmap = "gray", vmin = 0, vmax = 1)

# Expected_image
plt.figure()
plt.imshow((img.detach().cpu().numpy())[0, 0, :, :], cmap = "gray", vmin = 0, vmax = 1)


print("test size:  {}".format(test_size))
print("test_PSNR  :{}".format(20 * np.log10(1 / np.sqrt((32 * test_loss/test_size)))))

#-----------------------------image_demo----------------------------------------------------------------------------------------  
img = data_demo
img = torch.from_numpy(np.resize(img, (1, 1, img.shape[0], img.shape[1])))
img = img.to(device)
#-----------------Forward Pass----------------------
output = model(img)#, 16, "train")
# Reconstructed_image 
plt.figure()
plt.imshow((output.detach().cpu().numpy())[0, 0, :, :], cmap = "gray", vmin = 0, vmax = 1)
# Expected_image
plt.figure()
plt.imshow((img.detach().cpu().numpy())[0, 0, :, :], cmap = "gray", vmin = 0, vmax = 1)
