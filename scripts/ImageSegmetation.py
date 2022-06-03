from scripts.PreProcess import count_not_white_pixels
import cv2
import os
import glob
import random


def random_pach(image):
    """divide image using 'slide window' to patches ,each patch is 400x400 scale
    :parameter
    image : make as many patche as can from this image
    :returns
    images: list of all patches
    """
    height = len(image)  # y is height
    width = len(image[0])  # x is width
    y = random.randint(0, height//2)
    x = random.randint(0, width//2)
    div_im = image[y:y + 400, x:x + 400].copy()
    return div_im


def image_segment(image):
    """divide image using 'slide window' to patches ,each patch is 400x400 scale
    :parameter
    image : make as many patche as can from this image
    :returns
    images: list of all patches
    """
    height = len(image)  # y is height
    width = len(image[0])  # x is width
    y = 0
    images = []
    while y <= height - 400:
        x = 100
        while x < width - 200:
            div_im = image[y:y + 400, x:x + 400].copy()
            if count_not_white_pixels(div_im, 5000):
                images.append(div_im)
            x = x + 200
        y = y + 200

    return images


def make_segmetation(source_path, dest_path):
    """make patches from all images with in source path, and save them at destination path
    :parameter
    source_path : source path to images to make patches
    dest_path : save the patches in this directory
    """
    n_files = len(glob.glob1(source_path, "*.jpg"))
    f = 0
    for filename in os.listdir(source_path):
        f += 1
        print('\r {0} - Processing - {1} / {2}'.format(source_path, f, n_files), end='')
        image = cv2.imread(source_path + '/' + filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (1500, 1200))  ##########################לשים לב#################
        images = image_segment(image)
        for i in range(len(images)):
            cv2.imwrite(dest_path + '/' + filename.replace('.jpg', '_') + str(i + 1) + '.jpg', images[i])
