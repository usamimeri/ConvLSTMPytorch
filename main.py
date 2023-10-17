import numpy as np
import torch
from torch import nn
from convLSTM import ConvLSTM2d
from torch.utils.data import DataLoader
import imageio
from tqdm import tqdm
import os

# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/10/17 13:25:36
@Author  :   Zhang Wenjie 
@Version :   3.9.13
@Contact :   1710269958@qq.com
'''

# 128->64->64 5x5 the same as the paper


class Encode_Forecast(nn.Module):
    """
    Input:(B,F,S,H,W)
    this is a simplified version since the forecasting structure is omitted
    """

    def __init__(self, batch_size, img_size):
        super().__init__()
        self.encode_conv1 = ConvLSTM2d(
            1, 128, 5, batch_size, img_size, return_sequences=True)
        self.batchnorm3d_1 = nn.BatchNorm3d(128)
        self.encode_conv2 = ConvLSTM2d(
            128, 64, 5, batch_size, img_size, return_sequences=True)
        self.batchnorm3d_2 = nn.BatchNorm3d(64)
        self.encode_conv3=ConvLSTM2d(64,64,5,batch_size,img_size,return_sequences=True)
        self.batchnorm3d_3=nn.BatchNorm3d(64)
        self.conv_1x1 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x, _ = self.encode_conv1(x)
        x = self.batchnorm3d_1(x)
        x, _ = self.encode_conv2(x)
        x = self.batchnorm3d_2(x)
        x,_=self.encode_conv3(x)
        x=self.batchnorm3d_3(x)
        # the last time_step (B,F,H,W)->(B,C,H,W)
        x = self.conv_1x1(x[:, :, -1])
        x = torch.sigmoid(x)
        return x


os.makedirs('output_images',exist_ok=True)
moving_mnist = np.load('mnist_test_seq.npy')
moving_mnist = moving_mnist.transpose((1, 0, 2, 3))
#(B,S,H,W)
train = moving_mnist[:8000]
val = moving_mnist[8000:9000]
test = moving_mnist[9000:]


def collate(batch):
    batch = torch.tensor(np.array(batch)).unsqueeze(1)
    batch = batch / 255.0
    rand = np.random.randint(10, 20)
    return batch[:, :, rand-10:rand], batch[:, :, rand]

def collate_test(batch):
    target = np.array(batch)[:,10:]                     
    batch = torch.tensor(np.array(batch)).unsqueeze(1)          
    batch = batch / 255.0                             
    batch = batch.to(device)                          
    return batch, target


batch_size = 32
train_loader = DataLoader(
    train, shuffle=True, batch_size=batch_size, collate_fn=collate)
val_loader = DataLoader(
    val, shuffle=False, batch_size=batch_size, collate_fn=collate)
test_loader = DataLoader(
    test, shuffle=False, batch_size=batch_size, collate_fn=collate_test)


def save_images(epoch):
    # Output is the same as input (B,C,H,W)
    batch, target = next(iter(test_loader))

    output = np.zeros(target.shape, dtype=np.uint8)

    seq_len=target.shape[1]

    for time_step in range(seq_len):
        input_ = batch[:,:,time_step:time_step+10]   
        output[:,time_step]=(model(input_).squeeze(1).cpu()>0.5)*255.0 #(squeeze makes (B,1,C,H,W)->(B,C,H,W)) Binary Image

    
    for i in range(len(target)):
        # Write target video as gif
        imageio.mimsave(f"./output_images/{i}_target_epoch{epoch}.gif",target[i], "GIF", fps = 5)  
        # Write output video as gif
        imageio.mimsave(f"./output_images/{i}_output_epoch{epoch}.gif", output[i], "GIF", fps = 5)    
        

model = Encode_Forecast(batch_size, (64, 64))
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model = model.to(device)

for epoch in range(10):
    train_loss = 0
    num_samples = 0
    model.train()
    with tqdm(train_loader, desc="Train") as pbar:
        for data, label in pbar:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*len(data)
            num_samples += len(data)
            pbar.set_postfix({
                'Epoch': epoch+1,
                'Train Loss:': f"{train_loss/num_samples:.3f}",
            })
    val_loss = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        with tqdm(val_loader, desc="Validation") as pbar:
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                outout = model(data)
                loss = criterion(outout, label)
                val_loss += loss.item()*len(data)
                num_samples += len(data)
                pbar.set_postfix({
                    'Epoch': epoch+1,
                    'Train Loss:': f"{val_loss/num_samples:.3f}",
                })
    save_images(epoch)
                


torch.save(model.state_dict(), 'save.pt')