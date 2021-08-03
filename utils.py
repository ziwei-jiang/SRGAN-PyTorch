import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms 
from torchvision.models import vgg19
import numpy as np

class TV_Loss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		tv_height = torch.pow(x[:,:,1:,:] - x[:,:,:-1, :], 2).sum()
		tv_width = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2).sum()
		return (tv_height + tv_width)


class DIV2K_train_set(Dataset):
	def __init__(self, data_dir, upscale_factor=4, crop_size = 96):
		super().__init__()
		self.crop_size = crop_size
		self.hr_transform = transforms.Compose([
			transforms.RandomCrop(self.crop_size)
			])
		self.lr_transform = transforms.Compose([
			transforms.Resize(self.crop_size//upscale_factor, interpolation=Image.BICUBIC),
			transforms.ToTensor()
			])
		self.hr_normalize = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
			])

		self.file_names = [data_dir+file_name for file_name in os.listdir(data_dir)]

	def __len__(self):
		return len(self.file_names)

	def __getitem__(self, index):
		sample = Image.open(self.file_names[index])
		hr_crop = self.hr_transform(sample)
		lr_crop = self.lr_transform(hr_crop)
		hr_crop = self.hr_normalize(hr_crop)
		return hr_crop, lr_crop

class DIV2K_valid_set(Dataset):
	def __init__(self, data_dir, upscale_factor=4):
		super().__init__()
		self.upscale_factor = upscale_factor

		self.hr_normalize = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
			])
		self.file_names = [data_dir+file_name for file_name in os.listdir(data_dir)]

	def __len__(self):
		return len(self.file_names)

	def __getitem__(self, index):
		sample = Image.open(self.file_names[index])
		width, height = sample.size
		crop_size = int(np.min((width, height)))
		crop_size = crop_size - crop_size % self.upscale_factor
		hr_transform = transforms.Compose([
			transforms.CenterCrop(crop_size)
			])
		lr_transform = transforms.Compose([
			transforms.Resize(crop_size// self.upscale_factor, interpolation=Image.BICUBIC),
			transforms.ToTensor(),
			])
		hr_crop = hr_transform(sample)
		lr_crop = lr_transform(hr_crop)
		hr_crop = self.hr_normalize(hr_crop)
		return hr_crop, lr_crop


class FeatureExtractor(nn.Module):
	def __init__(self):
		super().__init__()
		vggnet = vgg19(pretrained=True)
		self.feature_extractor = nn.Sequential(*list(vggnet.features)[:36]).eval()
		for parameter in self.feature_extractor.parameters():
			parameter.requires_grad = False

	def forward(self, x):
		return self.feature_extractor(x)


