# RDN
import torch
from torch import nn

class RDN(nn.Module):
	def __init__(self, channels=3, scale_factor=2):
		super(RDN,self).__init__()
		self.D = 20		# RDB number 20
		self.C = 6		# number of conv layer in RDB 6
		self.G = 32		# growth rate 32
		self.G0 = 64	# local and global feature fusion layers 64filter

		kernel_size = 3
		in_channels = channels
		out_channels = channels
		scale_factor = scale_factor

		# net
		self.SFE_1 = nn.Conv2d(in_channels,self.G0, kernel_size=kernel_size, padding = kernel_size>>1, stride=1)
		self.SFE_2 = nn.Conv2d(self.G0, self.G0, kernel_size=kernel_size, padding=kernel_size>>1, stride=1 )
		

		self.RDBS = nn.ModuleList()
		for d in range(self.D):
			self.RDBS.append(RDB(self.G0,self.C,self.G,kernel_size))
		
		self.GFF = nn.Sequential(
			nn.Conv2d(self.D*self.G0, self.G0,kernel_size=1,padding=0,stride=1),
			nn.Conv2d(self.G0, self.G0,kernel_size=kernel_size,padding=kernel_size>>1, stride=1)
		)

		# up-sample
		assert 2 <= scale_factor <= 4
		if scale_factor % 2 == 0:
			up = []
			for i in range(scale_factor//2):
				up.extend([
					nn.Conv2d(self.G0,self.G0*(2**2), kernel_size=kernel_size, padding=3//2),
					nn.PixelShuffle(2)
				])
			self.UP = nn.Sequential(*up)
		else:
			self.UP = nn.Sequential(
				nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
				nn.PixelShuffle(scale_factor)
			)

		# self.UP = nn.Sequential(
		# 	nn.Conv2d(self.G0, self.G*4, kernel_size=kernel_size, padding=kernel_size>>1, stride=1),
		# 	nn.PixelShuffle(2),
		# 	nn.Conv2d(self.G,self.G*4, kernel_size=kernel_size, padding=kernel_size>>1, stride=1),
		# 	nn.PixelShuffle(2),
		# 	nn.Conv2d(self.G, out_channels, kernel_size=kernel_size, padding=kernel_size>>1, stride=1)
		# )

		self.OUT = nn.Conv2d(self.G0, out_channels, kernel_size=3, padding=3//2, stride=1)

		# init
		for p in self.modules():
			if isinstance(p, nn.Conv2d):
				nn.init.kaiming_normal(p.weight)
				if p.bias is not None:
					p.bias.data.zero_()
		return

	def forward(self, x):
		f1 = self.SFE_1(x)			# F_-1
		out = self.SFE_2(f1)		#	F_0

		RDB_outs = []
		for i in range(self.D):
			out = self.RDBS[i](out)		# F_d -> F_d(i)
			RDB_outs.append(out)
			# out = f1 + out	# what is this ???

		out = torch.cat(RDB_outs, 1)	# concat
		out = self.GFF(out)						# F_GF
		out = f1 + out								# F_DF
		out = self.UP(out)
		out = self.OUT(out)
		return out



class RDB(nn.Module):
	def __init__(self, G0, C, G, kernel_size=3):
		super(RDB,self).__init__()
		self.CONVS = nn.Sequential(*[DenseBlock(G0+i*G, G) for i in range(C)])
		self.LFF = nn.Conv2d(G0+C*G,G0,kernel_size=1,padding=0,stride=1)
		return
	def forward(self, x):
		out = self.CONVS(x)
		lff = self.LFF(out)
		return lff + x
			
class DenseBlock(nn.Module):
	def __init__(self,in_channels,out_channels,kernel_size = 3):
			super(DenseBlock,self).__init__()
			self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding = kernel_size>>1,stride= 1)
			self.relu = nn.ReLU()
	def forward(self,x):
			out = self.relu(self.conv(x))
			return torch.cat((x,out),1)

