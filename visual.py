import cv2
from datasets.pre_process import down_sample

import torch

path = '/share/dataset/coco/images/val2017/000000002153.jpg'

ckpt_path = '/data2/wangxinzhe/codes/github.com/xinzwang.cn/supre_resolution/checkpoints/RDN/2022-10-27_21-43-31/epoch=76_psnr=28.84595_ssim=0.89506_mse=0.00000.pt'

hr = cv2.imread(path) / 255.0

lr = down_sample(hr)

cv2.imwrite('img/lr.png', lr * 255.0)
cv2.imwrite('img/hr.png', hr * 255.0)


device= torch.device('cuda:0')

model = torch.load(ckpt_path)['model'].to(device)

with torch.no_grad():
	lr_ = torch.Tensor(lr.transpose(2,0,1)).unsqueeze(0).to(device)
	# hr_ = torch.Tensor(hr.transpose(2,0,1)).unsqueeze(0).to(device)
	pred_ = model(lr_)
	# pred2_ = model(hr_)

pred = pred_.cpu().numpy()[0].transpose(1,2,0)
# pred2 = pred2_.cpu().numpy()[0].transpose(1,2,0)

cv2.imwrite('img/pred_rdn.png', pred * 255)

# cv2.imwrite('img/pred_rdn_2.png', pred2 * 255)

