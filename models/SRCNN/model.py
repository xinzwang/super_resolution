from torch import nn

class SRCNN(nn.Module):
	def __init__(self,channels=3, scale_factor=2):
		super(SRCNN,self).__init__()
		self.up = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=True)
		self.conv = nn.Sequential (
			nn.Conv2d(channels,64,kernel_size=9,padding=9//2),
			nn.ReLU(inplace=True),
			nn.Conv2d(64,32,kernel_size=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32,channels ,kernel_size=5,padding=5//2)
		)

	def forward(self,x):
		print(x.shape)
		out = self.up(x)
		out = self.conv(out)
		return out