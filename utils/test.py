import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.autograd import Variable



def calc_psnr(sr, hr, scale=2, rgb_range=1.0, dataset=None):
	if hr.nelement() == 1: return 0
	
	diff = (sr - hr) / rgb_range

	if dataset and dataset.dataset.benchmark:
		shave = scale
		if diff.size(1) > 1:
				gray_coeffs = [65.738, 129.057, 25.064]
				convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
				diff = diff.mul(convert).sum(dim=1)
	else:
		shave =int(scale + 6)

	valid = diff[..., shave:-shave, shave:-shave]
	mse = valid.pow(2).mean()

	print(mse)

	return -10 * math.log10(mse)

def gaussian(window_size, sigma):
		gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
		return gauss/gauss.sum()

def create_window(window_size, channel):
		_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
		_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
		window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
		return window

def calc_ssim(img1, img2, window_size, channel, size_average = True):
		window = create_window(window_size, channel)
		mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
		mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

		mu1_sq = mu1.pow(2)
		mu2_sq = mu2.pow(2)
		mu1_mu2 = mu1*mu2

		sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
		sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
		sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

		C1 = 0.01**2
		C2 = 0.03**2

		ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

		if size_average:
				return ssim_map.mean()
		else:
				return ssim_map.mean(1).mean(1).mean(1)



def test(model, dataloader, device):
	psnr_list = []
	ssim_list = []
	for i,(lr, hr) in enumerate(tqdm(dataloader)):
		lr = lr.to(device)
		hr = hr.to(device)
		with torch.no_grad():
			pred = model(lr)
		psnr = calc_psnr(pred, hr)
		ssim = calc_ssim(pred, hr, window_size=11, channel=3)
		psnr_list.append(psnr)
		ssim_list.append(ssim_list)
	return np.mean(psnr_list), np.mean(ssim_list), 0