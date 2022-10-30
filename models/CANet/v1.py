import torch
import torch.nn as nn
from .common import ResBlock, default_conv, Upsampler
from .sft import SFTAffineLayer, SFTConcatLayer

class CANet(nn.Module):
	def __init__(self, channels=3, scale_factor=2, sft_mode='concat'):
		super(CANet, self).__init__()

		n_feats = 64
		knn_feats = 64

		n_LR_SFT = 6
		n_HR_SFT = 0

		if sft_mode == 'affine':
			sft = SFTAffineLayer
		elif sft_mode == 'concat':
			sft = SFTConcatLayer

		# knn feature
		self.conv_knn = nn.Sequential(
			# default_conv(in_channels=channels, out_channels=knn_feats, kernel_size=3),
			# default_conv(in_channels=channels, out_channels=knn_feats, kernel_size=3),
			ResBlock(conv=default_conv, n_feats=knn_feats, kernel_size=3),
			ResBlock(conv=default_conv, n_feats=knn_feats, kernel_size=3),
		)

		# knn upsample
		self.up_knn = nn.Upsample(scale_factor=scale_factor, mode='nearest')

		# head conv
		self.conv_head = nn.Sequential(
			default_conv(in_channels=channels, out_channels=knn_feats, kernel_size=3),
			ResBlock(conv=default_conv, n_feats=knn_feats, kernel_size=3),
		)
		
		# lr feature
		lr_barnch = []
		for i in range(n_LR_SFT):
			lr_barnch.append(sft(n_feats=n_feats,	knn_feats=knn_feats, hidden_feats=64, conv=default_conv))
		self.lr_branch = nn.Sequential(*lr_barnch)

		# upsample
		self.up = Upsampler(default_conv, scale_factor, n_feats)

		# hr feature
		hr_barnch = []
		for i in range(n_HR_SFT):
			hr_barnch.append(sft(n_feats=n_feats,	knn_feats=knn_feats, hidden_feats=64, conv=default_conv))
		self.hr_branch = nn.Sequential(*hr_barnch)

		# tail conv
		self.conv_tail = nn.Sequential(
			ResBlock(conv=default_conv, n_feats=knn_feats, kernel_size=3),
			default_conv(in_channels=n_feats, out_channels=channels, kernel_size=3)
		)
		return

	def forward(self, x):
		x1 = self.conv_head(x)

		lr_knn = self.conv_knn(x1)
		hr_knn = self.up_knn(lr_knn)

		x2, _ = self.lr_branch((x1, lr_knn))
		x3 = self.up(x2 + x1)
		x4, _ = self.hr_branch((x3, hr_knn))
		x5 = self.conv_tail(x4)

		return x5
