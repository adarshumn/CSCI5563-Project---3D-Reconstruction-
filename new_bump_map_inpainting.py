#NOTE: Imported from Google Colab and may not work exactly the same if you run. 
# Make sure to change the file paths if you try to run this

batch_size = 64
num_epochs = 20
wtl2 = 0.999
img_shapes = [256,256,3]

import os
import glob
import numpy as np
import random
import matplotlib.pyplot as plt 

import torch
import logging
from torch.utils.data import Dataset, DataLoader
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

class ImageFolderWithPaths(dset.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

data_transforms = transforms.Compose([transforms.ToTensor()])
bump_dataset = ImageFolderWithPaths('/content/drive/MyDrive/classes/spring_2021/csci_5563/celebahq_bump_data', transform=data_transforms)

bumpDataloader = torch.utils.data.DataLoader(bump_dataset, batch_size=batch_size, shuffle=True)

class Generator(torch.nn.Module):

    #Generator model
    def __init__(self):
        super(Generator,self).__init__()
        

        self.t1=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(4,4),stride=4,padding=1),
            torch.nn.LeakyReLU(0.2,inplace=True)
        )
        
        self.t2=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2,inplace=True)
        )
        self.t3=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(4,4),stride=2,padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2,inplace=True)
        )
        self.t4=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(4,4),stride=2,padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2,inplace=True)
        )
        self.t5=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(4,4),stride=2,padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2,inplace=True)
            
        )
        self.t6=torch.nn.Sequential(
            torch.nn.Conv2d(512,512,kernel_size=(4,4)),#bottleneck
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )
        self.t7=torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=(4,4),stride=2,padding=0),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
            )
        self.t8=torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=(4,4),stride=2,padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
            )
        self.t9=torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
            )
        self.t10=torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=(4,4),stride=4,padding=0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
            )
        self.t11=torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=(4,4),stride=4,padding=0, dilation=1),
            torch.nn.Tanh()
            )  
    
    def forward(self,x):
    	x=self.t1(x)#; print(x.shape)
    	x=self.t2(x)#; print(x.shape)
    	x=self.t3(x)#; print(x.shape)
    	x=self.t4(x)#; print(x.shape)
    	x=self.t5(x)#; print(x.shape)
    	x=self.t6(x)#; print(x.shape)
    	x=self.t7(x)#; print(x.shape)
    	x=self.t8(x)#; print(x.shape)
    	x=self.t9(x)#; print(x.shape)
    	x=self.t10(x)#; print(x.shape)
    	x=self.t11(x)#; print(x.shape)
    	return x #output of generator

class Discriminator(torch.nn.Module):

    #Discriminator model
    def __init__(self):
        super(Discriminator,self).__init__()
        
        self.t1=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(4,4),stride=4,padding=1),
            torch.nn.LeakyReLU(0.2,inplace=True)
        )
        
        self.t2=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(4,4),stride=4,padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2,inplace=True)
        )
        
        self.t3=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(4,4),stride=2,padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2,inplace=True)
        )
        self.t4=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(4,4),stride=2,padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2,inplace=True)
        )
        self.t5=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512,out_channels=1,kernel_size=(4,4),stride=1,padding=0),
            torch.nn.Sigmoid()
        )
            
    def forward(self,x):
      x=self.t1(x)#; print(x.shape)
      x=self.t2(x)#; print(x.shape)
      x=self.t3(x)#; print(x.shape)
      x=self.t4(x)#; print(x.shape)
      x=self.t5(x)#; print(x.shape)
      return x #output of discriminator

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(netG, netD, loss, batch_size, total_loss_history, recon_loss_history, adversarial_loss_history, disc_loss_history):

    # initialization on CPU
    input_real = torch.FloatTensor(batch_size, 3, 256, 256)
    input_cropped = torch.FloatTensor(batch_size, 3, 256, 256)
    label = torch.FloatTensor(batch_size)
    real_label = 1
    fake_label = 0

    # initialization on GPU
    # netD.cuda()
    # netG.cuda()
    # loss.cuda()
    # input_real, input_cropped,label = input_real.cuda(),input_cropped.cuda(), label.cuda()

    input_real = Variable(input_real)
    input_cropped = Variable(input_cropped)
    label = Variable(label)
    
    # Optimizers
    generator_optimizer = torch.optim.Adam(netG.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(netD.parameters(), lr=0.001)
    for epoch in range(num_epochs):
      i = 0
      for inputs, labels, paths in bumpDataloader:
        real_cpu = torch.FloatTensor(inputs)
        batch_size = real_cpu.size(0)
        with torch.no_grad():
            input_real.resize_(real_cpu.size()).copy_(real_cpu)
            input_cropped.resize_(real_cpu.size()).copy_(real_cpu)
            # random blocks
            xVals = random.sample(range(img_shapes[1]), 2)
            yVals = random.sample(range(img_shapes[0]), 2)
            xStart = min(xVals[0], xVals[1])
            xEnd = max(xVals[0], xVals[1])
            yStart = min(yVals[0], yVals[1])
            yEnd = max(yVals[0], yVals[1])
            print("mask width: " + str(xStart) + ":" + str(xEnd))
            print("mask height: " + str(yStart) + ":" + str(yEnd))
            input_cropped[:,:, yStart:yEnd, xStart:xEnd] = 255.0

        #start the discriminator by training with real data---
        netD.zero_grad()
        with torch.no_grad():
            label.resize_(batch_size).fill_(real_label)
        
        output = netD(input_real)
        output = torch.squeeze(output)
        disc_loss_real = loss(output, label) 
        #print("disc loss real: ", disc_loss_real)
        disc_loss_real.backward()
        D_x = output.data.mean()

        # train the discriminator with fake data---
        fake = netG(input_cropped)
        label.data.fill_(fake_label)
        output = netD(fake.detach())
        output = torch.squeeze(output)
        disc_loss_fake = loss(output, label)
        #print("disc loss fake: ", disc_loss_fake)
        disc_loss_fake.backward()
        D_G_z1 = output.data.mean()
        disc_loss = disc_loss_real + disc_loss_fake
        disc_loss_history.append(disc_loss)
        #print("disc loss total: ", disc_loss)
        discriminator_optimizer.step()

        #train the generator now---
        generator.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        output = torch.squeeze(output)
        gen_disc_loss = loss(output, label)
        adversarial_loss_history.append(gen_disc_loss)
        #print("gen disc loss: ", gen_disc_loss)

        wtl2Matrix = input_real.clone()
        wtl2Matrix.data.fill_(wtl2*10)
        wtl2Matrix.data[:,:, yStart:yEnd, xStart:xEnd] = wtl2

        gen_loss_l2 = (fake-input_real).pow(2)
        gen_loss_l2 = gen_loss_l2 * wtl2Matrix
        gen_loss_l2 = gen_loss_l2.mean()

        gen_loss = (1-wtl2) * gen_disc_loss + wtl2 * gen_loss_l2
        #print("gen loss: ", gen_loss)
        recon_loss_history.append(gen_loss_l2)
        total_loss_history.append(gen_loss)
        gen_loss.backward()

        D_G_z2 = output.data.mean()
        generator_optimizer.step()

        print('[%d / %d][%d / %d] Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f'
              % (epoch, num_epochs, i, len(bumpDataloader),
                 disc_loss.data, gen_disc_loss.data, gen_loss_l2.data, D_x, D_G_z1, ))
        i = i+1
        # if i == 1:
        #   print("SAVING IMAGES")
        #   vutils.save_image(input_real.data, '/content/drive/MyDrive/classes/spring_2021/csci_5563/results/real_samples_epoch_' + str(epoch) + '_' +str(i) +'.png')
        #   vutils.save_image(input_cropped.data, '/content/drive/MyDrive/classes/spring_2021/csci_5563/results/cropped_samples_epoch_' + str(epoch) + '_' +str(i) +'.png')
        #   recon_image = input_cropped.clone()
        #   recon_image.data[:,:, yStart:yEnd, xStart:xEnd] = fake.data[:,:, yStart:yEnd, xStart:xEnd]
        #   vutils.save_image(recon_image.data, '/content/drive/MyDrive/classes/spring_2021/csci_5563/results/recon_center_samples_epoch_' + str(epoch) + '_' +str(i) +'.png') 

#LETS TRAIN!
# Models
generator = Generator()
generator.apply(weights_init)
discriminator = Discriminator()
discriminator.apply(weights_init)

# loss
loss_function = torch.nn.BCELoss()

recon_loss_history = []
adversarial_loss_history = []
disc_loss_history = []
total_loss_history = []

train(generator, discriminator, loss_function, batch_size, total_loss_history, recon_loss_history, adversarial_loss_history, disc_loss_history)

fig, axes = plt.subplots(2,2, figsize=(15,15))
axes[0,0].plot(range(len(total_loss_history)), total_loss_history, color='blue')
axes[1,0].plot(range(len(recon_loss_history)), recon_loss_history, color='red')
axes[0,1].plot(range(len(adversarial_loss_history)), adversarial_loss_history, color='green')
axes[1,1].plot(range(len(disc_loss_history)), disc_loss_history, color='cyan')

axes[0,0].set_title("total loss")
axes[1,0].set_title("reconstruction loss")
axes[0,1].set_title("adversarial loss")
axes[1,1].set_title("discriminator loss")

plt.show()
fig.savefig('/content/drive/MyDrive/classes/spring_2021/csci_5563/results/loss_graph.png') 

#LETS USE IT!
test_dataset = ImageFolderWithPaths(root='/content/drive/MyDrive/classes/spring_2021/csci_5563/testing_input', transform=data_transforms)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True)
for inputs, labels, paths in test_dataloader:
  input_imgs = inputs
  new_imgs = generator(input_imgs)
  path = paths[0]
  num1 = path[(path.rfind("/") + 1):path.rfind(".")]
  path = paths[1]
  num2 = path[(path.rfind("/") + 1):path.rfind(".")]
  vutils.save_image(new_imgs, '/content/drive/MyDrive/classes/spring_2021/csci_5563/testing_output/network_rebuilt_' + num1 + "_" + num2 +'.png')