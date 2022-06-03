import cv2
import os
import glob
import random


def to_negativ(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = 255 - image
    return image


def make_negativ_dataset(source_path, dest_path):
    """make patches from all images with in source path, and save them at destination path
    :parameter
    source_path : source path to images to make patches
    dest_path : save the patches in this directory
    """
    os.makedirs(dest_path)
    n_files = len(glob.glob1(source_path, "*.jpg"))
    f = 0
    for filename in os.listdir(source_path):
        f += 1
        print('\r {0} - Processing - {1} / {2}'.format(source_path, f, n_files), end='')
        image = cv2.imread(source_path + '/' + filename)
        image = to_negativ(image)
        image = cv2.resize(image, (1500, 1200))  ##########################לשים לב#################
        cv2.imwrite(dest_path + '/' + filename, image)


make_negativ_dataset("../KHAT_splited/gender/test_gander/male", "../data_sets/negativ/KHAT/gender"
                                                         "/splited_test/male")
make_negativ_dataset("../KHAT_splited/gender/test_gander/female", "../data_sets/negativ/KHAT/gender"
                                                           "/splited_test/female")
