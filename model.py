import numpy as np
import torch
import torch.nn as nn


def convolution_block(in_channels, out_channels, stride):
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride),
		nn.BatchNorm2d(out_channels),
		nn.LeakyReLU(0.2)
		)

def upsample_block(in_channels):
	return nn.Sequential(
		nn.Conv2d(in_channels, in_channels*4, 3, padding=1),
		nn.PixelShuffle(2),
		nn.PReLU(in_channels)
		)

class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.PReLU(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = out + x
		return out

class Generator(nn.Module):
	def __init__(self, upscale_factor=4, num_blocks=16):
		super().__init__()
		num_upblocks = int(np.log2(upscale_factor))
		self.conv1 = nn.Conv2d(3, 64, 9, padding=4)
		self.relu = nn.PReLU(64)
		self.resblocks = nn.Sequential(*([ResidualBlock(64, 64)]* num_blocks))
		self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
		self.bn = nn.BatchNorm2d(64)
		self.upblocks = nn.Sequential(*([upsample_block(64)]* num_upblocks))
		self.conv3 = nn.Conv2d(64, 3, 9, padding=4)

	def forward(self, x):

		out = self.conv1(x)
		identity = self.relu(out)
		out = self.resblocks(identity)
		out = self.conv2(out)
		out = self.bn(out)
		out += identity
		out = self.upblocks(out)
		out = self.conv3(out)
		return torch.tanh(out)





class Discriminator(nn.Module):
	def __init__(self, crop_size = 128):
		super().__init__()

		num_ds = 4
		size_list = [64, 128, 128, 256, 256, 512, 512]
		stride_list = [1,2]*3
		self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
		self.relu = nn.LeakyReLU(0.2)
		self.convblocks = nn.Sequential(convolution_block(64, 64, 2), 
										*[convolution_block(in_ch, out_ch, stride)
										for in_ch, out_ch, stride in zip(size_list, size_list[1:], stride_list)])
		self.fc1 = nn.Linear(int(512*(crop_size/ 2**num_ds)**2), 1024)
		self.fc2 = nn.Linear(1024, 1)
		self.sig = nn.Sigmoid()

	def forward(self, x):
		out = self.conv1(x)
		out = self.relu(out)
		out = self.convblocks(out)
		out = self.fc1(out.view(out.size(0),-1))
		out = self.relu(out)
		out = self.fc2(out)
		out = self.sig(out)

		return out








# if __name__ == '__main__':
# 	dnet = Discriminator().cuda()
# 	gnet = Generator().cuda()

# 	summary(dnet, (3,96,96))
# 	summary(gnet,(3,24,24))



