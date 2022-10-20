import cv2
import glob
import numpy as np
from torch.utils.data import Dataset
from .pre_process import down_sample

class COCODataset(Dataset):
	def __init__(self, path, panel_size=128, scale_factor=2, test_flag=False):
		super().__init__()
		self.N = panel_size
		self.scale_factor = scale_factor
		self.img_paths = glob.glob(path + ('images/test2017/' if test_flag else 'images/val2017/') + '*.jpg')
		return

	def __len__(self):
		return len(self.img_paths)
	
	def __getitem__(self, index):
		p = self.img_paths[index]
		hr = cv2.imread(p).transpose(2, 0, 1).astype(np.float32)	# HWC->CHW, np.uint8->np.float32
		hr = hr / 255.0																						# [0-255]->[0,1]
		hr = hr[..., :self.N, :self.N]

		lr = down_sample(hr, scale_factor=self.scale_factor)
		return lr, hr

