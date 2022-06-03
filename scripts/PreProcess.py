from random import randint
import cv2
import numpy as np
import os
import math
import glob


def calc_angle(points_x, points_y):
    # Calculating the angle of the image rotation.
    # according to the squares parameters.
    if points_x[0] < points_x[1]:
        y = points_y[0]
        points_y[0] = points_y[1]
        points_y[1] = y

    a = abs(points_y[1] - points_y[0])
    b = abs(points_x[1] - points_x[0])
    c = math.sqrt(a * a + b * b)
    angle = math.acos(a / c)
    angle = 90 - math.degrees(angle)

    if points_y[1] > points_y[0]:
        return -angle
    else:
        return angle


def find_squares2(img_bgr):
    location = []
    img = img_bgr
    # img = cv2.imread('water_coins.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    marker_count, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    segmented = cv2.watershed(img, markers)

    # END of original watershed example

    output = np.zeros_like(img)
    output2 = img.copy()

    # Iterate over all non-background labels
    for i in range(2, marker_count + 1):
        mask = np.where(segmented == i, np.uint8(255), np.uint8(0))
        x, y, w, h = cv2.boundingRect(mask)
        area = cv2.countNonZero(mask[y:y + h, x:x + w])
        location.append([x, y])
        #print("Label %d at (%d, %d) size (%d x %d) area %d pixels" % (i, x, y, w, h, area))

        # Visualize
        color = randint(0, 255 + 1)
        output[mask != 0] = color
        cv2.rectangle(output2, (x, y), (x + w, y + h), color, 1)
        cv2.putText(output2, '%d' % i, (x + w // 4, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                    cv2.LINE_AA)
    # self.print_image(output)
    # self.print_image(output2)
    angle = calc_angle([location[0][0], location[1][0]], [location[0][1], location[1][1]])

    return angle


def rotate_image(img, angle):
    """ rotate image according to angle"""
    # Rotate a given image.
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    #print("rotate {0}ײ²ֲ° right".format(angle))
    return rotated


def remove_yellow(txt_img):
    """remove all yellow from given image
    :parameter
    txt_img : cropped image of the text part of the original image to remove yellow lines from text area
    :return
    the same image with no yellow color
    """
    ## (1) Read and convert to HSV
    hsv = cv2.cvtColor(txt_img, cv2.COLOR_BGR2HSV)

    ## (2) Find the target yellow color region in HSV
    hsv_lower = np.array([21, 39, 64])
    hsv_upper = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    ## (3) morph-op to remove horizone lines
    kernel = np.ones((5,1), np.uint8)
    mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    cv2.imwrite("mask2.jpg", mask2)
    src2 = cv2.imread('mask2.jpg', cv2.IMREAD_COLOR)

    #blend the images
    no_yellow_img = cv2.addWeighted(txt_img, 1, src2, 1, 0.0)

    return no_yellow_img


def crop_text(image):
    """crop text area from the original image
    :parameter
    image : original image
    :return
    image of text area only
    """
    # return text section as image
    y = len(image)  # y is height
    x = len(image[0]) #x is width
    return image[2000:4500, 370:x-470].copy()


def detect_circles(image):
    """detect circles in given image
    :parameter
    image : image to work on
    :return
    locations of found circles, None if not found
    """
    # detect if image has circles, if not returns None
    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im = cv2.medianBlur(im, 5)
    return cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, 1, 20,
                            param1=50, param2=30, minRadius=0, maxRadius=0)


def detect_gender(image):
    """detect the gender of the writer according to his mark
    :parameter
    image: the original image
    :return
    (str): the gender of the writer
    """
    # crop gender box from image
    #gender_image = image[790:970, 1820:2410].copy()
    gender_image = image[700:1300, 1720:2600].copy()
    gender_image = get_yellow_box(gender_image)
    y = len(gender_image)  # y is height
    x = len(gender_image[0])  # x is width

    # crop section of male
    male_img = gender_image[0:y, x//2: x].copy()
    # crop section of female
    female_img = gender_image[0:y, 0: x//2].copy()


    # check if male or female images has circles
    male_circles = detect_circles(male_img)
    female_circles = detect_circles(female_img)

    if female_circles is None and male_circles is not None:
        return 'male'
    elif female_circles is not None and male_circles is None:
        return 'female'
    else:
        return 'no circle'


def get_yellow_box(image):
    """detect the position of yellow box in image
    :parameter
    image: find yellow box in that image
     :return
     cropped image of the yellow box area only"""
    ## (1) Read and convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    ## (2) Find the target yellow color region in HSV
    hsv_lower = np.array([21, 39, 64])
    hsv_upper = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    ## (3) Find the max-area contour
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    ## (4) Crop and save it
    x, y, w, h = cv2.boundingRect(cnt)
    dst = image[y:y + h, x:x + w]
    return dst


def sort_in_folders(image, gender, filename):
    """sort and save images to the right folders
    :parameter
    image : image to sort and save
    gender : gender of the writer to sort to the right folder according to the gender
    filename: the name of the image
    """
    # Saves an image in the correct folder
    if gender == 'male':
        cv2.imwrite(r'gender\\male\\' + filename, image)
    elif gender == 'female':
        cv2.imwrite(r'gender\\female\\' + filename, image)
    elif gender == 'no circle':
        cv2.imwrite(r'gender\\no gender\\' + filename, image)


def count_not_white_pixels(im, black_px_min):
    """count how many not white pixels in given image
     :parameter
     im: image to count black pixels
     black_px_min: threshold of the minimum of black pixels
     :return
     (bool): true if there is more black pixels than the threshold, else false
    """
    black = 0
    for i in range(im.shape[0]):  # traverses through height of the image
        for j in range(im.shape[1]):  # traverses through width of the image
            if im[i][j] < 200:
                black += 1
    return black > black_px_min # return true if there is more black pixels than minimum black pixels

def check_handwrite_rows(im):
    """remove blank rows from the bottom of the text area
     :parameter
     im: remove blank rows from that image
     :returns
     image without blank rows at the bottom
     None: if the whole image is blank rows"""
    # remove empty rows from image
    y = len(im)  # y is height
    x = len(im[0])  # x is width
    h = 0
    while count_not_white_pixels(im[h:h+100,0:x], 500) and h < y:
        h += 100
    im = im[0:h,0:len(im[0])].copy()
    return im if h > 0 else None


def proc_image(filename):
    """process the original image to get the gender and text area only and sort it to the correct folder
    :parameter
    filename:name of the image to process
    """
    image = cv2.imread('images\\' + filename)
    # resize image
    image = cv2.resize(image, (4816, 6843))
    # Align the image
    angle = find_squares2(image)
    image = rotate_image(image, angle)
    #cv2.imwrite('ProcessedImage\\' + filename, image)  # save rotated image if u want to check

    # detect if image was written by male or female
    gender = detect_gender(image)

    # get text area only
    text_img = crop_text(image)
    # remove yellow lines
    text_img = remove_yellow(text_img)
    # grayscale text image
    text_img = cv2.resize(text_img, (1500, 1200))
    text_img = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)

    #crop hand writing text only
    text_img = check_handwrite_rows(text_img)
    if text_img is not None:
        sort_in_folders(text_img, gender, filename)


if __name__ == '__main__':
    directory = 'images'
    n_files = len(glob.glob1('images',"*.jpg"))
    i = 0
    for filename in os.listdir(directory):
        i += 1
        print('\r Processing {0} / {1}'.format(i, n_files), end='')
        if filename.endswith(".jpg"):
            proc_image(filename)





