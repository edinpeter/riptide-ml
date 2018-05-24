import cv2
from make_dice import alpha_image, add_alpha, random_blur, brightness
import numpy as np
import os
import random

die_snap_count = 0

def get_backgrounds():
	backgrounds_dir = 'backgrounds'
	backgrounds = os.listdir(backgrounds_dir)
	background_images = []
	for background in backgrounds:
		background_images.append(cv2.imread(os.path.join(backgrounds_dir, background), cv2.IMREAD_UNCHANGED))
	return background_images

def add_dice(bg_img, die_img, die_number):
	global die_snap_count

	if bg_img.shape[2] < 4:
		bg_img = add_alpha(bg_img)
	size = 100

	dice_snap_dir = 'dice_snaps'

	die_img = cv2.resize(die_img, (size, size))
	alpha_bg = alpha_image(bg_img.shape[0:2], die_img.dtype)

	bg_img_width = bg_img.shape[1]
	bg_img_height = bg_img.shape[0]

	edge_factor = size / 2 + 5
	die_center_x = random.randint(edge_factor, bg_img_width - edge_factor)
	die_center_y = random.randint(edge_factor, bg_img_height - edge_factor)
	
	half_size = size / 2
	alpha_bg[die_center_y - half_size : die_center_y + half_size, die_center_x -  half_size : die_center_x +  half_size] = die_img

	alpha_mask = cv2.inRange(alpha_bg, (0,0,0,255), (255,255,255,255))
	alpha_mask_not = cv2.bitwise_not(alpha_mask)

	bg_comp = cv2.bitwise_and(bg_img, bg_img, mask=alpha_mask_not)
	alpha_comp = cv2.bitwise_and(alpha_bg, alpha_bg, mask=alpha_mask)

	comp = cv2.bitwise_or(bg_comp, alpha_comp)

	die_with_bg = comp[die_center_y - half_size : die_center_y + half_size, die_center_x -  half_size : die_center_x +  half_size]
	print die_with_bg.shape

	cv2.imwrite(os.path.join(dice_snap_dir, die_number + '_' + str(die_snap_count) + '.png'), die_with_bg)
	die_snap_count += 1
	return comp

def random_die():
	dice_dir = 'color_nbg'
	dice_names = os.listdir(dice_dir)
	die = dice_names[random.randint(0, len(dice_names))]
	die_img = cv2.imread(os.path.join(dice_dir, die), cv2.IMREAD_UNCHANGED)
	return die_img, die[0]

if __name__ == '__main__':
	dest = 'samples'
	bg_imgs = get_backgrounds()
	for q in range(len(bg_imgs)):
		bg = bg_imgs[q]
		for i in range(0, random.randint(1, 4)):
			die, number = random_die()
			die = random_blur(die)
			bg = add_dice(bg, die, number)
		cv2.imwrite(os.path.join(dest, str(q) + '.png'), bg)

