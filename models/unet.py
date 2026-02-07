"""
1D UNET model to learn time profile from the BrdU concentration
"""

import torch.nn as nn
import torch


class Unet(nn.Module):

    def __init__(self,kernel_size:int=3,pool_kernel_size:int=2,channels:list[int]=[64,128,256,512,1024],kernel_size_upconv:int=2):
        super().__init__()

        # Contractive Path
        self.down_block1 = DownBlock(in_channel=1,out_channel=channels[0],kernel_size=kernel_size)
        self.down_block2 = DownBlock(in_channel=channels[0],out_channel=channels[1],kernel_size=kernel_size)
        self.down_block3 = DownBlock(in_channel=channels[1],out_channel=channels[2],kernel_size=kernel_size)
        self.down_block4 = DownBlock(in_channel=channels[2],out_channel=channels[3],kernel_size=kernel_size)

        # Double convolution at the bottom of the U
        self.doubleconv = nn.Sequential(
            nn.Conv1d(in_channels=channels[3],out_channels=channels[4],kernel_size=kernel_size), 
            nn.ReLU(),
            nn.Conv1d(in_channels=channels[4],out_channels=channels[4],kernel_size=kernel_size),
            nn.ReLU(),
        )
                
        # Expansive Path
        self.up_block1 = UpBlock(in_channel=channels[4],out_channel=channels[3],kernel_size_upconv=kernel_size_upconv,kernel_size=kernel_size)
        self.up_block2 = UpBlock(in_channel=channels[3],out_channel=channels[2],kernel_size_upconv=kernel_size_upconv,kernel_size=kernel_size)
        self.up_block3 = UpBlock(in_channel=channels[2],out_channel=channels[1],kernel_size_upconv=kernel_size_upconv,kernel_size=kernel_size)
        self.up_block4 = UpBlock(in_channel=channels[1],out_channel=channels[0],kernel_size_upconv=kernel_size_upconv,kernel_size=kernel_size)

        # Last Layer
        self.classif = nn.Conv1d(in_channels=channels[0],out_channels=6,kernel_size=1)
        
    def forward(self,x):
        # Contractive
        down1,p1 = self.down_block1(x)
        down2,p2 = self.down_block2(p1)
        down3,p3 = self.down_block3(p2)
        down4,p4 = self.down_block4(p3)

        # Bottom
        x = self.doubleconv(p4)

        # Expansive
        x = self.up_block1(x,down4)
        x = self.up_block2(x,down3)
        x = self.up_block3(x,down2)
        x = self.up_block4(x,down1)

        return self.classif(x)

class DownBlock(nn.Module):
    def __init__(self,in_channel:int,out_channel:int,kernel_size:int=3,pool_kernel_size:int=2,pool_stride:int=2):
        super().__init__()

        self.stack = nn.Sequential(
            nn.Conv1d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size), 
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channel,out_channels=out_channel,kernel_size=kernel_size),
            nn.ReLU(),
            )
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size,stride=pool_stride)
        
    def forward(self,x):
        down = self.stack(x)
        p = self.pool(down)
        return down,p

class UpBlock(nn.Module):
    def __init__(self,in_channel:int,out_channel:int,kernel_size_upconv:int=2,kernel_size:int=3):
        super().__init__()
        self.convT = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channel,out_channels=in_channel//2, kernel_size=kernel_size_upconv,stride=2)
        )
        self.stack = nn.Sequential(
            nn.Conv1d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size), 
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channel,out_channels=out_channel,kernel_size=kernel_size),
            nn.ReLU(),
        )

    def forward(self,x1,x2): # To copy and crop the down feature map
        x1 = self.convT(x1)

        #TODO test this code
        diff = x2.size()[2] - x1.size()[2]
        x2_cropped = x2[:, :, diff//2 : x2.size()[2]-diff//2]
        if x2_cropped.size()[2] != x1.size()[2]:
             x2_cropped = x2[:, :, diff//2 : diff//2 + x1.size()[2]]

        x = torch.cat((x2_cropped,x1),dim=1)
        return self.stack(x)
        
