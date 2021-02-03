import time
import os, os.path
import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable


import KSUS
import mnist_reader
from data_processing import get_data


#-------------------------prerequisite-----------------------------------------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
dir_path = os.path.dirname(os.path.realpath(__file__))


#-------------------------data_preprocessing-----------------------------------------------------------------------------------------

train_size = 48#000
test_size = 10#00

data_train, data_test, data_demo = get_data(train_size, test_size)

h, w = data_train.shape[1], data_train.shape[2]
#-----------------------------dataset----------------------------------------------------------------------------------------

class ksus_Dataset(Dataset):
  def __init__(self,datasetclean):
    self.clean=datasetclean
  
  def __len__(self):
    return len(self.clean)
  
  def __getitem__(self,idx):
    xClean=self.clean[idx]
    return xClean 
 
trainset = ksus_Dataset(data_train)
trainloader = DataLoader(trainset, batch_size = 32, shuffle = True)
testset = ksus_Dataset(data_test)
testloader = DataLoader(testset, batch_size = 32, shuffle = True)


#-----------------------------model_establishment----------------------------------------------------------------------------------------
#mask_mode = "varied" : mask_type = "1d"(not working) / "2d"
#mask_mode = "fixed" : mask_type = "1drandom" / "2drandom"
mask_mode = "fixed" 
mask_type = "1duniformc"
sparsity = 0.1
lr = 0.001
model = KSUS.ksus_model(mask_mode = mask_mode, mask_type = mask_type, sparsity = sparsity, h = h, w = w)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
model.to(device)

#-----------------------------train_procedure----------------------------------------------------------------------------------------

epochs = 79

for e in range(epochs):
    epoch = e + 1
    epochloss = 0
    t_start = time.time()
    print("Entering Epoch: ", epoch)
    for i, data in enumerate(trainloader, 0):
        running_loss = 0
        img = data
        img = img.view(img.shape[0], 1, img.shape[1], img.shape[2]).type(torch.FloatTensor)
        img = img.to(device)
    #-----------------Forward Pass----------------------
        output = model(img, 2 ** np.int(1.8 * np.sqrt(epoch)) , "train") #np.int(epoch/20) np.int(1.8 * np.sqrt(epoch))
        loss=criterion(output, img)
    #-----------------Backward Pass---------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 300 == 0:
            print(running_loss)
        
        epochloss += loss.item()
  #-----------------Log-------------------------------
    t_end = time.time()
    print("======> epoch: {}/{}, PSNR:{}, loss:{}".format(epoch,epochs, 20 * np.log10(1 / np.sqrt(32 * epochloss / train_size)), np.sqrt(epochloss / train_size)))
    print(" time: {}s".format(t_end - t_start))
    


# Sampled_mask
free_weights = np.fft.fftshift(np.copy(model.mask_layer.weights.detach().cpu().numpy()[0, 0, :, :]))
plt.figure()
plt.imshow(free_weights, cmap = "gray", vmin = 0, vmax = 1)
plt.figure()
plt.hist(np.resize(free_weights,(h * w)), bins = 100)

# Reconstructed_image
plt.figure()
plt.imshow((output.detach().cpu().numpy())[0, 0, :, :], cmap = "gray", vmin = 0, vmax = 1)

# Expected_image
plt.figure()
plt.imshow((img.detach().cpu().numpy())[0, 0, :, :], cmap = "gray", vmin = 0, vmax = 1)

torch.save(model.state_dict(), dir_path + '/model/' + mask_mode + '_' +  mask_type + '_' + '1.pt')

#-----------------------------test_procedure----------------------------------------------------------------------------------------    
# if model is needed to be loaded
#model2 = KSUS.ksus_model(mask_mode = mask_mode, mask_type = mask_type, sparsity = sparsity, h = h, w = w)
#model.to(device)
#model2.load_state_dict(torch.load(dir_path + '/model/' + mask_mode + '_' +  mask_type + '_' + '1.pt'))
#model2.eval()

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

print("train size: {}".format(train_size))
print("test size:  {}".format(test_size))
print("mask type:  {}".format(mask_type))
print("mask mode:  {}".format(mask_mode))
print("sparsity:   {}".format(sparsity))
print("epoch: {} & learning rate: {}".format(epochs, lr))
print("Train_PSNR: {}".format(20 * np.log10(1 / np.sqrt(32 * epochloss/train_size))))
print("Test_PSNR  :{}".format(20 * np.log10(1 / np.sqrt((32 * test_loss/test_size)))))

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