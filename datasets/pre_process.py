import cv2


def down_sample(x, scale_factor):
	out = cv2.resize(x, (0,0), fx=1 / scale_factor, fy=1 / scale_factor, interpolation=cv2.INTER_CUBIC)
	return out
	