import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import numpy as np
import os, os.path
import mnist_reader
import U_net
from torch.autograd import Variable

#imgs = []
#path = os.path.abspath(os.getcwd()) + "/Test_image/"
#valid_images = [".jpg"]
#for f in os.listdir(path):
    #ext = os.path.splitext(f)[1]
    #if ext.lower() not in valid_images:
     #   continue
    #imgs.append(Image.open(os.path.join(path,f)))

train_size = 50
test_size = 5
total_size = train_size + test_size
x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='t10k')
x_train = x_train[0 : total_size, :]
xdata = np.resize(x_train, (total_size, 28, 28))
xdata = np.pad(xdata, ((0,0), (2,2), (2,2)), 'constant')  # get to 32x32
xdata = xdata / 255
#plt.imshow(xdata[19, :, :])
np.random.seed(0)
noise = 0.5 * np.random.rand(total_size, 32, 32)
xdata_com = np.clip(noise + xdata - 0.25, 0, 1)
xdata = xdata.astype('float64') 
xdata_com = xdata_com.astype('float64') 
xdata_test = xdata[train_size: total_size]
xdata_com_test = xdata_com[train_size: total_size]
xdata = xdata[0:train_size]
xdata_com = xdata_com[0:train_size]


class noisedDataset(Dataset):
  def __init__(self,datasetnoised,datasetclean):
    self.noise=datasetnoised
    self.clean=datasetclean
  
  def __len__(self):
    return len(self.noise)
  
  def __getitem__(self,idx):
    xNoise=self.noise[idx]
    xClean=self.clean[idx]
    return (xNoise, xClean) 
 
trainset = noisedDataset(xdata_com, xdata)
trainloader = DataLoader(trainset, batch_size = 32, shuffle = True)
testset = noisedDataset(xdata_com_test, xdata_test)
testloader = DataLoader(testset, batch_size = 1, shuffle = True)

model = U_net.UNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)


#We check whether cuda is available and choose device accordingly
#if torch.cuda.is_available()==True:
#  device="cuda:0"
#else:
#  device ="cpu"



epochs = 5
l=len(trainloader)
losslist=list()
epochloss=0
running_loss=0




for epoch in range(epochs + 1):
    print("Entering Epoch: ", epoch)
    for i, data in enumerate(trainloader, 0):
        dirty, clean = data
        dirty=dirty.view(dirty.shape[0], 1, dirty.shape[1], dirty.shape[2]).type(torch.FloatTensor)
        clean=clean.view(clean.shape[0], 1, clean.shape[1], clean.shape[2]).type(torch.FloatTensor)

    #-----------------Forward Pass----------------------
        output=model(dirty)
        loss=criterion(output,clean)
    #-----------------Backward Pass---------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        if i % 20 == 0:
            print(running_loss)
        running_loss = 0
        epochloss+=loss.item()
  #-----------------Log-------------------------------
    losslist.append(running_loss/l)
    print("======> epoch: {}/{}, Loss:{}".format(epoch,epochs,epochloss))
    epochloss = 0

test_loss = 0
for i, data in enumerate(testloader, 0):
    dirty, clean = data
    dirty=dirty.view(dirty.shape[0], 1, dirty.shape[1], dirty.shape[2]).type(torch.FloatTensor)
    clean=clean.view(clean.shape[0], 1, clean.shape[1], clean.shape[2]).type(torch.FloatTensor)

    #-----------------Forward Pass----------------------
    output=model(dirty)
    loss=criterion(output,clean)
    test_loss += loss.item()
print(test_loss/50)

plt.figure()
plt.imshow((dirty.detach().numpy())[0, 0, :, :], cmap = "gray", vmin=0, vmax=1)
plt.figure()
plt.imshow((output.detach().numpy())[0, 0, :, :], cmap = "gray", vmin=0, vmax=1)
plt.figure()
plt.imshow((clean.detach().numpy())[0, 0, :, :], cmap = "gray", vmin=0, vmax=1)
