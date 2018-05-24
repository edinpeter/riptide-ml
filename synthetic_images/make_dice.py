import os
import cv2
import numpy as np
import sys
from multiprocessing import Process
import random

multi = True
def random_blur(image):
    blur_amt = random.randint(0,10)
    blur_amt = blur_amt if blur_amt % 2 == 1 else blur_amt + 1
    return cv2.blur(image, (blur_amt, blur_amt))

def alpha_image(dims, dtype):
    alpha_c = np.ones(dims, dtype=dtype) * 255
    alpha_a = np.zeros(dims, dtype=dtype)
    alpha = cv2.merge((alpha_c, alpha_c, alpha_c, alpha_a))
    return alpha

def add_alpha(image):
    b, g, r = cv2.split(image)
    alpha = np.ones(b.shape, dtype=b.dtype) * 255
    return cv2.merge((b, g, r, alpha))

def shift_image(image):
    alpha = (255, 255, 255, 0)
    im_shape = image.shape

    #vertical
    vertical_shift = random.randint(-im_shape[0] / 3, im_shape[0] / 3)
    image = np.roll(image, vertical_shift, axis=0)
    if vertical_shift < 0:
        image[im_shape[0] + vertical_shift : im_shape[0], 0 : im_shape[1]] = alpha
    else:
        image[0 : vertical_shift, 0 : im_shape[1]] = alpha

    #horizontal
    horizontal_shift = random.randint(-im_shape[1] / 3, im_shape[1] / 3)
    image = np.roll(image, horizontal_shift, axis=1)

    if horizontal_shift < 0:
        image[0 : im_shape[0], im_shape[1] + horizontal_shift : im_shape[1]] = alpha
    else:
        image[0 : im_shape[0] - 1, 0 : horizontal_shift] = alpha
    return image

def brightness(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    brighten_amount = random.randint(-100, 100)
    if brighten_amount > 0:
        v = np.where(v >= 255 - brighten_amount, 255, v + brighten_amount)
    else:
        v = np.where(v <= 0 - brighten_amount, 0, v + brighten_amount)
    v = v.astype(h.dtype)
    hsv_image = cv2.merge((h, s, v))

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def strip_red_background(image):
    channels = cv2.split(image)
    red_low = (0, 0, 160, 255)
    red_high = (130,130,255,255)

    mask = cv2.inRange(image, red_low, red_high)

    alpha = alpha_image(channels[0].shape, channels[0].dtype)

    mask_not = cv2.bitwise_not(mask)

    orig_comp = cv2.bitwise_and(image, image, mask=mask_not)    
    alpha_comp = cv2.bitwise_and(alpha, alpha, mask=mask)

    return cv2.bitwise_or(orig_comp, alpha_comp)

def add_noise(image):
    channels = cv2.split(image)
    alpha = (255, 255, 255, 0)
    mask = cv2.inRange(image, alpha, alpha)
    mask_not = cv2.bitwise_not(mask)
    noise = np.random.rand(image.shape[0], image.shape[1], image.shape[2]) * 255
    noise_a = np.ones((image.shape[0], image.shape[1]), dtype=channels[0].dtype) * 255
    noise = noise.astype(channels[0].dtype)
    noise_c = cv2.split(noise)
    noise = cv2.merge((noise_c[0], noise_c[1], noise_c[2], noise_a))

    orig_comp = cv2.bitwise_and(image, image, mask=mask_not)
    noise_comp = cv2.bitwise_and(noise, noise, mask=mask)

    return cv2.bitwise_or(orig_comp, noise_comp)

def make_grayscale(class_num, doall=True):
    color_dir = 'color'
    gray_dir = 'gray'
    for image in os.listdir(color_dir):
        if doall or str(class_num) + '_' in image:
            img_gray = cv2.imread(os.path.join(color_dir, image), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(gray_dir, image), img_gray)

def remove_background(class_num, doall=True):
    color_dir = 'color'
    color_nbg_dir = 'color_nbg'
    for image in os.listdir(color_dir):
        if doall or str(class_num) + '_' in image:
            img = cv2.imread(os.path.join(color_dir, image), cv2.IMREAD_UNCHANGED)
            img = add_alpha(img)
            img = strip_red_background(img)
            cv2.imwrite(os.path.join(color_nbg_dir, image), img)

def make_nbg_grays(class_num, doall=True):
    gray_nbg_dir = 'gray_nbg'
    color_nbg_dir = 'color_nbg'
    for image in os.listdir(color_nbg_dir):
        if doall or str(class_num) + '_' in image:
            img = cv2.imread(os.path.join(color_nbg_dir, image), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(gray_nbg_dir, image), img)

def make_noisy(class_num, doall=True):
    color_nbg_dir = 'color_nbg'
    color_noise_dir = 'color_noise'
    for image in os.listdir(color_nbg_dir):
        if doall or str(class_num) + '_' in image:
            img = cv2.imread(os.path.join(color_nbg_dir, image), cv2.IMREAD_UNCHANGED)
            img = add_noise(img)
            cv2.imwrite(os.path.join(color_noise_dir, image), img)

def make_gray_noisy(class_num, doall=True):
    gray_noise_dir = 'gray_noise'
    color_noise_dir = 'color_noise'
    for image in os.listdir(color_noise_dir):
        if doall or str(class_num) + '_' in image:
            img = cv2.imread(os.path.join(color_noise_dir, image), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(gray_noise_dir, image), img)

def make_shifted(class_num, doall=True):
    color_nbg_dir = 'color_nbg'
    color_nbg_shift_dir = 'color_shift'
    for image in os.listdir(color_nbg_dir):
        if doall or str(class_num) + '_' in image:
            img = cv2.imread(os.path.join(color_nbg_dir, image), cv2.IMREAD_UNCHANGED)
            img = shift_image(img)
            cv2.imwrite(os.path.join(color_nbg_shift_dir, image), img)

def make_shift_noisy(class_num, doall=True):
    color_nbg_shift_dir = 'color_shift'
    color_shift_noise_dir = 'color_shift_noise'
    for image in os.listdir(color_nbg_shift_dir):
        if doall or str(class_num) + '_' in image:
            img = cv2.imread(os.path.join(color_nbg_shift_dir, image), cv2.IMREAD_UNCHANGED)
            img = add_noise(img)
            img = brightness(img)

            img = random_blur(img)
            cv2.imwrite(os.path.join(color_shift_noise_dir, image), img)

def make_threads(function):
    threads = []
    for i in range(0,7):
        t = Process(target=function, args=(i, False,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

def make_images(i, image_count):
    import cube_maker as cubes
    cubes.make_dice(i, image_count)


if __name__ == "__main__":
    operations = [make_grayscale,
                  remove_background,
                  make_shifted,
                  make_nbg_grays,
                  make_noisy,
                  make_gray_noisy,
                  #make_images,
                  make_shift_noisy]

    image_count = 30
    if len(sys.argv) > 1:
           image_count = int(sys.argv[1])

    print "Making images..."
    if make_images in operations:
    	for i in range(1, 7):
    		make_images(i, image_count)

    print "Making grayscale..."
    if multi and make_grayscale in operations:
        make_threads(make_grayscale)
    elif make_grayscale in operations:
        make_grayscale(0)
    
    print "Removing backgrounds..."
    if multi and remove_background in operations:
        make_threads(remove_background)
    elif remove_background in operations:
        remove_background(0)    

    print "Removing backgrounds from grays..."
    if multi and make_nbg_grays in operations:
        make_threads(make_nbg_grays)
    elif make_nbg_grays in operations:
        make_nbg_grays(0)

    print "Adding noise backgrounds..."
    if multi and make_noisy in operations:
        make_threads(make_noisy)
    elif make_noisy in operations:
        make_noisy(0)
        
    print "Adding gray noise..."
    if multi and make_gray_noisy in operations:
        make_threads(make_gray_noisy)
    elif make_gray_noisy in operations:
        make_gray_noisy(0)

    print "Shifting images..."
    if multi and make_shifted in operations:
        make_threads(make_shifted)
    elif make_shifted in operations:
        make_shifted(0)

    print "Noisying shifted images..."
    if multi and make_shift_noisy in operations:
        make_threads(make_shift_noisy)
    elif make_shift_noisy in operations:
        make_shift_noisy(0)
        

