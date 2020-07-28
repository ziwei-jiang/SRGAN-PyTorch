import argparse
import os
import numpy as np
from model import Generator
from PIL import Image
import torch
import torchvision.transforms as transforms 



parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, help='input image')
parser.add_argument('--upscale_factor', type=int, default=4, choices=[2,4,8], help='super resolution upscale factor')
parser.add_argument('--weight', type=str, help='generator weight file')
parser.add_argument('--downsample', type=str, default=None, choices=[None, 'bicubic'], help='Downsample the input image before applying SR')
parser.add_argument('--cuda', action='store_true', help='Using GPU to run')


if __name__ == '__main__':
	opt = parser.parse_args()
	upscale_factor = opt.upscale_factor
	input_img = opt.image
	weight = opt.weight
	out_dir = 'results/'+input_img[input_img.rfind('/')+1:input_img.find('.')]
	os.makedirs(out_dir, exist_ok=True)

	if not torch.cuda.is_available() and opt.cuda:
		raise Exception("No GPU available")
	with torch.no_grad():
		generator_net = Generator(upscale_factor = upscale_factor, num_blocks=16).eval()
		

		saved_G_weight = torch.load(weight) 
		generator_net.load_state_dict(saved_G_weight)


		img = Image.open(input_img)
		img_format = img.format
		if opt.downsample == 'bicubic':
			size = np.min(img.size)
			downscale = transforms.Resize(int(size//upscale_factor), interpolation=Image.BICUBIC)
			img = downscale(img)

		img_tensor = transforms.ToTensor()(img).unsqueeze(0)
		if torch.cuda.is_available() and opt.cuda:
			img_tensor = img_tensor.cuda()
			generator_net.cuda()

		sr_tensor = generator_net(img_tensor)
		
		sr_transform = transforms.Compose([
			transforms.Normalize((-1,-1,-1),(2,2,2)),
			transforms.ToPILImage()
			])


		sr_img = sr_transform(sr_tensor[0].data.cpu())


		sr_img.save(out_dir+'/sr_' + input_img[input_img.rfind('/')+1:])

		w, h = img.size 
		w *= upscale_factor
		h *= upscale_factor

		upscale = transforms.Resize((h,w), interpolation=Image.BICUBIC)
		lr_img = upscale(img)
		lr_img.save(out_dir+'/bicubic_' + input_img[input_img.rfind('/')+1:])
		img.save(out_dir+'/lr_'+input_img[input_img.rfind('/')+1:])






