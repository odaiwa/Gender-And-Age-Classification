import cv2
import os
#from pdf2image import convert_from_path

tiff_base_path = "./TIFF/"
pdf_base_path = "./PDF"
new_path = "./JPG/"
pdf_to_jpg_path = "./PDF_TO_JPG/"
def tif_to_jpg():
    i = 0
    for tiff in os.listdir(tiff_base_path):
        i += 1
        print("file : " + tiff)
        read = cv2.imread(tiff_base_path + tiff)
        print(tiff.split('.')[0])
        outfile = tiff.split('.')[0] + '.jpg'
        cv2.imwrite(new_path + outfile, read, [int(cv2.IMWRITE_JPEG_QUALITY), 200])
    print('{0} tiff files has been processed'.format(i))
tif_to_jpg()