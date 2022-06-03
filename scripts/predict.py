import numpy as np
import cv2
from GenderClassification import GenderClassification
import os
from scripts.ImageSegmetation import image_segment, random_pach
import glob
import cv2
from PIL import Image
import torchvision
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,classification_report

FEMALE = 0
MALE = 1
test_female_path = "../KATT_FULL_DATA/test/female"
test_male_path = "../KATT_FULL_DATA/test/male"
models_path = "../new_KHATT_iz_1000_-71.73333333333333.h5"


def to_negativ(x: Image):
    im_np = np.asarray(x)
    im_np = 255 - im_np
    im_pil = Image.fromarray(im_np)
    return im_pil


transform = torchvision.transforms.Compose([torchvision.transforms.Resize(448),
                                            torchvision.transforms.CenterCrop(448),
                                            torchvision.transforms.Lambda(to_negativ),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize([0.4856, 0.4994, 0.4324],
                                                                             [0.1817, 0.1811, 0.1927])])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def predict_image(im_name, model):
    """predict the gender of the writer, extract patches from the image and process them for the model
    :parameter
    im_name: hand writing image name or path to predict gender
    model: use this model to predict image
    :return
    0: if the gender is Female
    1: if the gender is Male
    """
    image = cv2.imread(im_name,0)
    patches = image_segment(image)
    if len(patches) % 2 == 0:
        patches.append(random_pach(image))
    x_test = []
    for patch in patches:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        image = Image.fromarray(patch)
        img_tensor = transform(image).to(device).unsqueeze(0)
        x_test.append(img_tensor)
    # print(im_name + " : ", end='\r')
    predictions = model.predict(x_test)
    score = [0, 0]
    for prediction in predictions:
        score[prediction[0]] = score[prediction[0]] + 1

    '''print(
        f'male score: {score[1]} \nfemale score :{score[0]}\nclassification : {"MALE" if score[1] > score[0] else "FEMALE"}',
        end='\r')'''
    return MALE if score[1] > score[0] else FEMALE


def predect_image_folder_by_model(male_folder, female_folder, model_path):
    print("=========================================================================================")
    print("summary :")
    print(f"model: {model_path}")
    print(f"dataset : {male_folder}")
    ganderClassification = GenderClassification(device=device, modelPath=model_path)
    clsses = []
    predections = []
    for image_name in os.listdir(female_folder):
        clsses.append(FEMALE)
        predections.append(predict_image(female_folder + '/' + image_name, ganderClassification))


    m_predicted = 0
    for image_name in os.listdir(male_folder):
        clsses.append(MALE)
        predections.append(predict_image(male_folder + '/' + image_name, ganderClassification))
    tn, fp, fn, tp = confusion_matrix(clsses, predections).ravel()
    print("summary :")
    print(f" {tp} {fp} \n {fn} {fn}")
    print(f"model: {model_path}")
    print(f"dataset : {male_folder}")
    print('female predicted : {0} \\ {1}'.format(len(list(filter(lambda x: x == FEMALE, predections))), len(list(filter(lambda x: x == FEMALE, clsses)))))
    print('male predicted : {0} \\ {1}'.format(len(list(filter(lambda x: x == MALE, predections))), len(list(filter(lambda x: x == MALE, clsses)))))
    print(f'Accuracy : {accuracy_score(clsses, predections):.2f}%')
    #print(classification_report(np.array(clsses), np.array(predections), labels=['female','male']))


if __name__ == '__main__':
    # for model_path in os.listdir(models_path):

    predect_image_folder_by_model(test_male_path, test_female_path, models_path)


def predict_image2(image, model, preprocess_func, scale):
    """predict the gender of the writer, extract patches from the image and process them for the model
    :parameter
    im_name: hand writing image name or path to predict gender
    model: use this model to predict image
    preprocess_func: the model pre process function to make the patches fit to model
    scale : scale for resize each patch to fit model
    :return
    (float) Gender probability to be male
    (float) Gender probability to be female
    (int) number of patches extracted form the image
    (str) final gender classification: Male or Female
    """
    patches = image_segment(image)  # extract patches from image
    # if len(patches) % 2 == 0: patches.append(rand_patch(image)) # if number of patches is even, add 1 random patch to make it odd
    x_test = []
    for patch in patches:  # Process each patch to fit the model
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        patch = 255 - patch
        patch = cv2.resize(patch, scale)
        patch = preprocess_func(patch)
        x_test.append(patch)

    patches_count = len(x_test)
    print('number of patches extracted from image : ', patches_count)
    predictions = model.predict(np.asarray(x_test))

    m_score, f_score = 0, 0  # count how many patches was classify to each class
    for prediction in predictions:
        if prediction[0] > 0.5:
            f_score = f_score + 1
        else:
            m_score = m_score + 1

    print('male score : ', m_score / patches_count)  # get final classification
    print('female score : ', f_score / patches_count)
    print('classification : {0}'.format('MALE' if m_score > f_score else 'FEMALE'))
    print()
    return m_score / patches_count, f_score / patches_count, patches_count, (
        '{0}'.format('MALE' if m_score > f_score else 'FEMALE'))


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
    return black > black_px_min  # return true if there is more black pixels than minimum black pixels


def check_image_size(image):
    """resize image to proper resolution before making patches"""
    y = len(image)  # y is height
    x = len(image[0])  # x is width
    print(x, y)
    if y > 3000 or x > 3000:
        new_x = int(x * (2 / 3))
        new_y = int(y * (2 / 3))
        image = cv2.resize(image, (new_y, new_x))
    elif y > 2500 or x > 2500:
        new_x = int(x * (2.3 / 3))
        new_y = int(y * (2.3 / 3))
        image = cv2.resize(image, (new_y, new_x))
    print(len(image[0]), len(image))
    return image
