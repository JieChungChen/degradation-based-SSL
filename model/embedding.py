import torch
import torch.nn as nn
from itertools import repeat


class SpatialDropout(nn.Module):
    def __init__(self, drop=0.16):
        super(SpatialDropout, self).__init__()
        self.drop = drop
        
    def forward(self, inputs, noise_shape=None):
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim()-2), inputs.shape[-1])   # 默认沿着中间所有的shape
        
        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)    
            outputs.mul_(noises)
            return outputs
            
    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)
    
    
class FCEmbeddingLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FCEmbeddingLayer, self).__init__()
        self.linear = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        b, n, c, l = x.shape
        x = x.reshape(b, n, c*l)
        out = self.linear(x)
        return out
    

def conv_block(in_ch, out_ch, kernel_size, padding, activation=True):
    if activation:
        return nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                            nn.ReLU())
    else:
        return nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size, padding),)
    

class CNNEmbeddingLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CNNEmbeddingLayer, self).__init__()
        self.conv1_1 = conv_block(in_ch, 32, kernel_size=5, padding=2)
        self.conv1_2 = conv_block(32, 32, kernel_size=5, padding=2)
        self.conv1_3 = conv_block(32, 32, kernel_size=5, padding=2)
        self.conv1_4 = conv_block(32, 32, kernel_size=5, padding=2)
        self.maxpool1_1 = nn.MaxPool1d(2, 2)
        self.maxpool1_2 = nn.MaxPool1d(2, 2)
        self.maxpool1_3 = nn.MaxPool1d(2, 2)
        self.spatial_drop1 = SpatialDropout(0.16)
        self.conv2_1 = conv_block(64, 32, kernel_size=11, padding=5)
        self.conv2_2 = conv_block(32, 32, kernel_size=11, padding=5)
        self.conv2_3 = conv_block(32, 64, kernel_size=7, padding=3)
        self.conv2_4 = conv_block(64, 64, kernel_size=7, padding=3)
        self.maxpool2_1 = nn.MaxPool1d(2, 2)
        self.maxpool2_2 = nn.MaxPool1d(2, 2)
        self.avgpool1 = nn.AvgPool1d(2, 2)
        self.conv3_1 = conv_block(96, 256, kernel_size=7, padding=3)
        self.conv3_2 = conv_block(256, 256, kernel_size=7, padding=3)
        self.conv3_3 = conv_block(256, 256, kernel_size=5, padding=2)
        self.conv3_4 = conv_block(256, 256, kernel_size=5, padding=2)
        self.spatial_drop2 = SpatialDropout(0.16)
        self.gloavgpool = nn.AdaptiveAvgPool1d(1)
        self.glomaxpool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(512, out_ch)

    def forward(self, x):
        b, n, c, l = x.shape
        x = x.reshape(b*n, c, l)
        conv1_1 = self.conv1_1(x)
        conv1_1 = self.conv1_2(conv1_1)
        conv1_2 = self.conv1_3(conv1_1)
        conv1_2 = self.conv1_4(conv1_2)
        conv1_out = self.maxpool1_3(torch.cat((self.maxpool1_1(conv1_1), self.maxpool1_2(conv1_2)), dim=1))
        conv1_out = self.spatial_drop1(conv1_out)
        conv2_1 = self.conv2_1(conv1_out)
        conv2_1 = self.conv2_2(conv2_1)
        conv2_2 = self.conv2_3(conv2_1)
        conv2_2 = self.conv2_4(conv2_2)
        conv2_out = self.avgpool1(torch.cat((self.maxpool2_1(conv2_1), self.maxpool2_2(conv2_2)), dim=1))
        conv3_1 = self.conv3_1(conv2_out)
        conv3_1 = self.conv3_2(conv3_1)
        conv3_2 = self.conv3_3(conv3_1)
        conv3_2 = self.conv3_4(conv3_2)
        conv3_out = self.spatial_drop2(torch.cat((conv3_1, conv3_2), dim=1))
        conv3_out = torch.add(self.gloavgpool(conv3_out), self.glomaxpool(conv3_out))
        out = self.linear(torch.squeeze(conv3_out))
        out = out.reshape(b, n, -1)
        return out