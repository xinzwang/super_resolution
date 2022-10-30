import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import models
from datasets.COCO import COCODataset
from utils.logger import build_logger
from utils.test import test, infer_img


def parse_args():
		parser = argparse.ArgumentParser()
		parser.add_argument('--model', default='SRCNN', help='SR model')
		parser.add_argument('--dataset', default='/share/dataset/coco/')
		parser.add_argument('--scale_factor', default=2, type=int)
		parser.add_argument('--panel_size', default=128, type=int)
		parser.add_argument('--batch_size', default=64)
		parser.add_argument('--epoch', default=5001)
		parser.add_argument('--lr', default=1e-4, type=float, help='Learning Rate')
		parser.add_argument('--seed', default=17, type=int)
		parser.add_argument('--device', default='cuda:1')
		args = parser.parse_args()
		print(args)
		return args


def build_dataset(args, test_flag=False):
		dataset = COCODataset(
				args.dataset,
				panel_size=args.panel_size,
				scale_factor=args.scale_factor,
				test_flag=test_flag
		)
		dataloader = DataLoader(
				dataset=dataset,
				batch_size=args.batch_size,
				num_workers=20,
				shuffle=(not test_flag),  # shuffle only train
		)
		return dataset, dataloader


def train(args):
		t = time.strftime('%Y-%m-%d_%H-%M-%S')
		ckpt_path = 'checkpoints/%s/%s/' % (args.model, t)
		if not os.path.exists(ckpt_path):
				os.makedirs(ckpt_path)
		log_path = 'log/%s/' % (args.model)
		if not os.path.exists(log_path):
				os.makedirs(log_path)
		logger = build_logger(log_path + '%s.log' % (t))

		logger.info(str(args))

		# seed
		torch.manual_seed(args.seed)
		torch.cuda.manual_seed(args.seed)

		# device
		device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

		# dataset
		dataset, dataloader = build_dataset(args)
		test_dataset, test_dataloader = build_dataset(args, test_flag=True)

		# model
		model = getattr(models, args.model)(
				channels=3, scale_factor=args.scale_factor)

		# 模型转换到相应设备上
		model.to(device)

		# loss optim
		loss_fn = nn.L1Loss()
		optimizer = optim.Adam(model.parameters(), args.lr)
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(
				optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, min_lr=1e-5)
		
		infer_img(ckpt_path + 'init/', num=9, model=model, dataloader=test_dataloader,device=device)	
 
		# train loop
		for epoch in range(args.epoch):
				total_loss = []
				print('epoch:%d lr:%f' % (epoch, optimizer.param_groups[0]['lr']))
				logger.info('epoch:%d lr:%f' %
										(epoch, optimizer.param_groups[0]['lr']))
				for i, (lr, hr) in enumerate(tqdm(dataloader)):
						lr = lr.to(device)
						hr = hr.to(device)
						optimizer.zero_grad()
						pred = model(lr)
						loss = loss_fn(pred, hr)
						loss.backward()
						optimizer.step()
						total_loss.append(loss.item())
						if i % 20 == 0:
								logger.info('epoch:%d batch:%d loss:%.5f' %
														(epoch, i, loss.item()))
				mean_loss = np.mean(total_loss)
				scheduler.step(mean_loss)
				# test
				psnr, ssim, mse = test(model, test_dataloader, device)
				infer_img(ckpt_path + 'epoch=%d_psnr=%.5f_ssim=%.5f_mse=%.5f/' % (epoch, psnr, ssim, mse), num=9, model=model, dataloader=test_dataloader,device=device)	
				# log
				logger.info('epoch:%d mean_loss:%.5f psnr=%.5f ssim=%.5f mse=%.5f' % (
						epoch, mean_loss, psnr, ssim, mse))
				# save ckpt
				save_path = ckpt_path + \
						'epoch=%d_psnr=%.5f_ssim=%.5f_mse=%.5f.pt' % (epoch, psnr, ssim, mse)
				torch.save({
						'modelname': args.model,
						'model': model,
						'dataset': args.dataset
				}, save_path)
		# finally, save the last model
		psnr, ssim, mse = test(model, test_dataloader, device)
		logger.info('epoch:%d mean_loss:%.5f psnr=%.5f ssim=%.5f mse=%.5f' %
								(epoch, mean_loss, psnr, ssim, mse))
		save_path = ckpt_path + \
				'final_epoch=%d_psnr=%.5f_ssim=%.5f_mse=%.5f.pt' % (psnr, ssim, mse)
		torch.save({
				'modelname': args.model,
				'model': model,
				'dataset': args.dataset
		}, save_path)
		return


if __name__ == '__main__':
		args = parse_args()
		train(args)
