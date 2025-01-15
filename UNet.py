import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensorV2

class TrainingData():
    def __init__(self):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.train_path = 'stage1_train/'
        self.folders = os.listdir(self.train_path)
        self.witdh = 128
        self.height = 128
        self.length = len(self.folders)
        
        self.img_list = []
        self.mask_list = []
        
        self.transforms = self.get_transforms(0.5, 0.5)

        # Read all training data
        for i in range(10):
            print("%d/%d"%(i , self.length))
            img , mask = self.get_item(i)
            self.img_list.append(img)
            self.mask_list.append(mask)
            
        self.img_list = [np.array(img) for img in self.img_list]  # 確保所有元素是 NumPy 陣列
        self.img_list = torch.FloatTensor(np.array(self.img_list)).to(self.device)
        self.mask_list = torch.FloatTensor(np.array(self.mask_list)).to(self.device)
        
        self.img_list = torch.permute(self.img_list , (0 , 3 , 1 , 2)) # 128x128x3 ->3x128x128 
        self.mask_list = torch.permute(self.mask_list , (0 , 3 , 1 , 2))
        
    def get_transforms(self, mean, std):
        list_transforms = []
            
        list_transforms.extend(
                    [
                HorizontalFlip(p=0.5), # only horizontal flip as of now
                    ])
        list_transforms.extend(
                    [
        Normalize(mean=mean, std=std, p=1),
        ToTensorV2(),
                    ])
        list_trfms = Compose(list_transforms)
        return list_trfms
    
    def get_item(self , index):
        
        image_folder = os.path.join(self.train_path , self.folders[index],'images/')
        mask_folder = os.path.join(self.train_path , self.folders[index],'masks/')
        image_path = os.path.join(image_folder,os.listdir(image_folder)[0])
        img = cv2.imread(image_path)[:,:,:3].astype('float32')
        img = cv2.resize(img, (self.height, self.witdh), interpolation=cv2.INTER_AREA)
        img /= 255.0

        mask = self.get_mask(mask_folder)
        #augmented = self.transforms(image = img, mask = mask)
        #img = augmented['image']
        #mask = augmented['mask']
        
        return img , mask
        
    def get_mask(self , mask_folder):
        
        mask = np.zeros((self.height, self.witdh, 1))

        for mask_ in os.listdir(mask_folder):
            mask_ = cv2.imread(os.path.join(mask_folder,mask_))
            mask_ = cv2.resize(mask_, (self.height, self.witdh), interpolation=cv2.INTER_AREA)
            mask_ = (cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY) > 0 ) * 1.0 
            mask_ = np.expand_dims(mask_,axis=-1)
            mask = np.maximum(mask, mask_)
        return mask
    
class TestingData():
    def __init__(self):
        self.test_path = 'stage1_test/'
        self.folders = os.listdir(self.train_path)
        self.witdh = 256
        self.height = 256
        
        self.img_list = []
        self.mask_list = []
        # Read all training data
        for i in range(len(self.folders)):
            img , mask = self.get_item(i)
            self.img_list.append(img)
            self.mask_list.append(mask)
            print(img.shape)
        self.img_list = np.array(self.img_list).to(self.device)
        self.mask_list = np.array(self.mask_list).to(self.device)
        
        
    def get_item(self , index):
        
        image_folder = os.path.join(self.test_path , self.folders[index],'images/')
        image_path = os.path.join(image_folder,os.listdir(image_folder)[0])
        img = cv2.imread(image_path)[:,:,:3].astype('float32')        
        return img 
    
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # encoder group 1
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, padding=1) # input_channel, output_channel, kernel_size, strid
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, padding=1) 
        
        # encoder group 2
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, padding=1) 
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, padding=1) 
        
        # encoder group 3
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, padding=1) 
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, padding=1) 
        
        # encoder group 4
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, padding=1) 
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, padding=1) 
        
        # middle 
        self.conv_mid = nn.Conv2d(512, 1, 3, 1, padding=1) 

        # decoder group 1
        self.conv_tf1 = nn.ConvTranspose2d(1, 512, 2, stride = 2) # input_channel, output_channel, kernel_size , stride
        self.conv5_1 = nn.Conv2d(1024, 512, 3, 1, padding=1) 
        self.conv5_2 = nn.Conv2d(512, 1, 3, 1, padding=1) 
        
        # decoder group 2
        self.conv_tf2 = nn.ConvTranspose2d(1, 256, 2, stride = 2) # input_channel, output_channel, kernel_size , stride
        self.conv6_1 = nn.Conv2d(512, 256, 3, 1, padding=1) 
        self.conv6_2 = nn.Conv2d(256, 1, 3, 1, padding=1) 
        
        # decoder group 3
        self.conv_tf3 = nn.ConvTranspose2d(1, 128, 2, stride = 2) # input_channel, output_channel, kernel_size , stride
        self.conv7_1 = nn.Conv2d(256, 128, 3, 1, padding=1) 
        self.conv7_2 = nn.Conv2d(128, 1, 3, 1, padding=1) 
        
        # decoder group 3
        self.conv_tf4 = nn.ConvTranspose2d(1, 64, 2, stride = 2) # input_channel, output_channel, kernel_size , stride
        self.conv8_1 = nn.Conv2d(128, 64, 3, 1, padding=1) 
        self.conv8_2 = nn.Conv2d(64, 64, 3, 1, padding=1) 
        
        self.conv_out= nn.Conv2d(64, 1, 1, 1) # input_channel, output_channel, kernel_size , stride
        
        
    def forward(self, X):
        
        # encoder group 1
        X = F.relu(self.conv1_1(X) , inplace=True) # 3 -> 64
        Conv1 = F.relu(self.conv1_2(X) , inplace=True) # 64 -> 64
        X = F.max_pool2d(Conv1, 2, 2)
        
        # encoder group 2
        X = F.relu(self.conv2_1(X) , inplace=True) # 64 -> 64
        Conv2 = F.relu(self.conv2_2(X) , inplace=True) # 64 -> 128
        X = F.max_pool2d(Conv2, 2, 2) 
        
        # encoder group 3
        X = F.relu(self.conv3_1(X) , inplace=True) # 128 -> 256
        Conv3 = F.relu(self.conv3_2(X) , inplace=True)# 256 -> 256
        X = F.max_pool2d(Conv3, 2, 2) 
        
        # encoder group 4
        X = F.relu(self.conv4_1(X) , inplace=True) # 256 -> 512
        Conv4 = F.relu(self.conv4_2(X) , inplace=True) # 512 -> 512
        X = F.max_pool2d(Conv4, 2, 2) 
        
        # middle
        X = F.relu(self.conv_mid(X) , inplace=True) # 512 -> 1

        # decoder group 1
        X = self.conv_tf1(X) # 1 -> 512
        X = torch.cat([X, Conv4], dim=1) # Concatenates 512 + 512 = 1024
        X = F.relu(self.conv5_1(X) , inplace=True) # 1024 -> 512
        X = F.relu(self.conv5_2(X) , inplace=True) # 512 -> 1
        
        # decoder group 2
        X = self.conv_tf2(X) # 1 -> 256
        X = torch.cat([X, Conv3], dim=1) # Concatenates 256 + 256 = 512
        X = F.relu(self.conv6_1(X) , inplace=True) # 512 -> 256
        X = F.relu(self.conv6_2(X) , inplace=True) # 256 -> 1       
        
        # decoder group 3
        X = self.conv_tf3(X) # 1 -> 128
        X = torch.cat([X, Conv2], dim=1) # Concatenates 128 + 128 = 256
        X = F.relu(self.conv7_1(X) , inplace=True) # 256 -> 128
        X = F.relu(self.conv7_2(X) , inplace=True) # 128 -> 1       
        
        # decoder group 4
        X = self.conv_tf4(X) # 1 -> 64
        X = torch.cat([X, Conv1], dim=1) # Concatenates 64 + 64 = 128
        X = F.relu(self.conv8_1(X) , inplace=True) # 128 -> 64
        X = F.relu(self.conv8_2(X) , inplace=True) # 64 -> 64     
        X = self.conv_out(X) # 64 -> 1
        #X = F.sigmoid(self.conv_out(X)) # 64 -> 1
        
        
        return X
    
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets,  epsilon=1e-6 , smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE =  BCE + dice_loss
        return Dice_BCE
    
    
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
          
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        TP = (inputs * targets).sum()      
        print(TP)                      
        FP = inputs.sum() - TP
        FN = targets - inputs.sum()
        IoU = TP / (TP + FP + FN)
        return 1 - IoU
    
class Main():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.training_data = TrainingData()
        print(self.training_data.img_list.shape)
        print(self.training_data.mask_list.shape)
        #print(self.training_data.img_list.max())
        #print(self.training_data.img_list.min())
        #print(self.training_data.mask_list.max())
        #print(self.training_data.mask_list.min())
        #print(self.training_data.mask_list[2,0])
        
        self.data_length = self.training_data.img_list.shape[0]
        self.batch_size = 10
        self.epochs = 5000
        self.lr = 0.00003
        self.loss_function = DiceBCELoss().to(self.device)
        self.U_Net = UNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.U_Net.parameters(), lr = self.lr)
        
        self.loss_list = []
        # check batch size
        if self.batch_size > self.data_length:
            self.batch_size = self.data_length
            
        print(self.U_Net)
        
        self.train()
        self.verify()
                
        
    def verify(self):
        for i in range(self.data_length):
            index = np.random.choice(self.data_length , 1 , replace=False)

            with torch.no_grad():
                
                # image process
                output = self.U_Net(self.training_data.img_list[index])
                output = torch.squeeze(output.detach().cpu())
                output = F.sigmoid(output)
                output = output * 0.5 + 0.5
                #output = output.clip(0 , 1)
                #print(output)
                #output = (output > 0.5) * 0.1
                
                print(output)
                
                org_image = torch.squeeze(self.training_data.img_list[index].detach().cpu())
                org_image = torch.permute(org_image , (1 , 2 , 0))
                
                target = torch.squeeze(self.training_data.mask_list[index].detach().cpu())
                print(output.shape)
                print(target.shape)
                
                plt.figure(1)
                plt.title('Loss')
                plt.plot(self.loss_list)
                plt.figure(2)
                plt.subplot(131)
                plt.title('Actual Image')
                plt.imshow(org_image , cmap='gray')
                plt.subplot(132)
                plt.title('Predicted Mask')
                plt.imshow(output, cmap='gray')
                plt.subplot(133)
                plt.title('Actual Mask')
                plt.imshow(target , cmap='gray')
                plt.show()
                
    def mask_convert(self , mask):
        
        print(mask.shape)

        mask = torch.squeeze(mask.detach().cpu())
        print(mask.shape)

        mask = (mask > 0.5) * 1.0
        print(mask.shape)

        return mask.view(128,128)
    
    
    def train(self):
        for epoch in range(self.epochs):
            index = np.random.choice(self.data_length , self.batch_size , replace=False)
            output = self.U_Net(self.training_data.img_list[index])
            loss = self.loss_function(output, self.training_data.mask_list[index])
            
            loss.backward()
            self.optimizer.step()
            if epoch % 5 ==0:
                print('[{}/{}] Loss:'.format(epoch+1, self.epochs), loss.item())
            self.loss_list.append(loss.item())
            
    
        
if __name__ == '__main__':
    main = Main()