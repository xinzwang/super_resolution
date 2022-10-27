import cv2


def down_sample(x, scale_factor=2, kernel_size=(9, 9), sigma=3):
	out = cv2.GaussianBlur(x, ksize=kernel_size, sigmaX=sigma, sigmaY=sigma)
	out = cv2.resize(out, (0,0), fx=1 / scale_factor, fy=1 / scale_factor, interpolation=cv2.INTER_CUBIC)
	return out


if __name__=='__main__':
	import cv2
	img = cv2.imread('/share/dataset/coco/val2017/000000581781.jpg') / 255.0

	cv2.imwrite('hr.png', img * 255.0)
	cv2.imwrite('lr.png', down_sample(img,2)*255.0)


	