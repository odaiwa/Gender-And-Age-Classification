from scripts.PreProcess import count_not_white_pixels
import cv2
import os
import glob


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
        #image = 255 - image #negative
        image = cv2.resize(image, (1500, 1200))  ##########################לשים לב#################
        images = image_segment(image)
        for i in range(len(images)):
            cv2.imwrite(dest_path + '/' + filename.replace('.jpg', '_') + str(i + 1) + '.jpg', images[i])


make_segmetation('/home/languagedetection/Datasets/gender-age/quwi-1/arabic/train/male','/home/languagedetection/Datasets/gender-age/quwi-1/segmented/arabic/train/male')
make_segmetation('/home/languagedetection/Datasets/gender-age/quwi-1/arabic/train/female','/home/languagedetection/Datasets/gender-age/quwi-1/segmented/arabic/train/female')
make_segmetation('/home/languagedetection/Datasets/gender-age/quwi-1/arabic/test/male','/home/languagedetection/Datasets/gender-age/quwi-1/segmented/arabic/test/male')
make_segmetation('/home/languagedetection/Datasets/gender-age/quwi-1/arabic/test/female','/home/languagedetection/Datasets/gender-age/quwi-1/segmented/arabic/test/female')
make_segmetation('/home/languagedetection/Datasets/gender-age/quwi-1/arabic/valid/male','/home/languagedetection/Datasets/gender-age/quwi-1/segmented/arabic/valid/male')
make_segmetation('/home/languagedetection/Datasets/gender-age/quwi-1/arabic/valid/female','/home/languagedetection/Datasets/gender-age/quwi-1/segmented/arabic/valid/female')



make_segmetation('/home/languagedetection/Datasets/gender-age/quwi-1/english/train/male','/home/languagedetection/Datasets/gender-age/quwi-1/segmented/english/train/male')
make_segmetation('/home/languagedetection/Datasets/gender-age/quwi-1/english/train/female','/home/languagedetection/Datasets/gender-age/quwi-1/segmented/english/train/female')
make_segmetation('/home/languagedetection/Datasets/gender-age/quwi-1/english/test/male','/home/languagedetection/Datasets/gender-age/quwi-1/segmented/english/test/male')
make_segmetation('/home/languagedetection/Datasets/gender-age/quwi-1/english/test/female','/home/languagedetection/Datasets/gender-age/quwi-1/segmented/english/test/female')
make_segmetation('/home/languagedetection/Datasets/gender-age/quwi-1/english/valid/male','/home/languagedetection/Datasets/gender-age/quwi-1/segmented/english/valid/male')
make_segmetation('/home/languagedetection/Datasets/gender-age/quwi-1/english/valid/female','/home/languagedetection/Datasets/gender-age/quwi-1/segmented/english/valid/female')



make_segmetation('/home/languagedetection/Datasets/gender-age/quwi-1/mix/train/male','/home/languagedetection/Datasets/gender-age/quwi-1/segmented/mix/train/male')
make_segmetation('/home/languagedetection/Datasets/gender-age/quwi-1/mix/train/female','/home/languagedetection/Datasets/gender-age/quwi-1/segmented/mix/train/female')
make_segmetation('/home/languagedetection/Datasets/gender-age/quwi-1/mix/test/male','/home/languagedetection/Datasets/gender-age/quwi-1/segmented/mix/test/male')
make_segmetation('/home/languagedetection/Datasets/gender-age/quwi-1/mix/test/female','/home/languagedetection/Datasets/gender-age/quwi-1/segmented/mix/test/female')
make_segmetation('/home/languagedetection/Datasets/gender-age/quwi-1/mix/valid/male','/home/languagedetection/Datasets/gender-age/quwi-1/segmented/mix/valid/male')
make_segmetation('/home/languagedetection/Datasets/gender-age/quwi-1/mix/valid/female','/home/languagedetection/Datasets/gender-age/quwi-1/segmented/mix/valid/female')
