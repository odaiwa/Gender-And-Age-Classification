import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
from scripts.ImageSegmetation import image_segment
import tensorflow as tf
import glob
from random import randint
from scripts.PreProcess import count_not_white_pixels

FEMALE = 0
MALE = 1

model1_name = 'models/vgg16_gender_split_segmentation_12ep.h5'
model2_name = 'models/vgg16_gender_split_segmentation_19ep.h5'
model3_name = 'models/vgg19_gender_split_segmentation_18ep.h5'

test_female_path = r'gender_split\test\female'
test_male_path = r'gender_split\test\male'


def predict_image(im_name, model1, model2, model3):
    image = cv2.imread(im_name, 0)
    patches = image_segment(image)
    if len(patches) % 2 == 0: # if patches is even number add 1 more random patch
        patches.append(random_crop(image))
    print(im_name + " : ")
    print('number of patches : ', len(patches))
    print()

    res1 = get_prediction(model1, 'vgg16', patches)
    res2 = get_prediction(model2, 'vgg16', patches)
    res3 = get_prediction(model3, 'vgg19', patches)
    print('final classification : {0}'.format('MALE' if sum([res1, res2, res3]) > 1.5 else 'FEMALE'))
    print()
    print('------------------------------------------------------------')
    return MALE if sum([res1, res2, res3]) > 1.5 else FEMALE


def get_prediction(model, model_type ,patches):
    print('model : {0}'.format(model_type))
    proc_patches = preprocess_patches(patches, model_type)
    predictions = model.predict(np.asarray(proc_patches))
    return get_score(predictions)


def preprocess_patches(patches, model_type):
    x_test = []
    if model_type == 'vgg16':
        for patch in patches:
            patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
            patch = cv2.resize(patch, (224, 224))
            patch = tf.keras.applications.vgg16.preprocess_input(patch)
            x_test.append(patch)
        return x_test

    elif model_type == 'vgg19':
        for patch in patches:
            patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
            patch = cv2.resize(patch, (224, 224))
            patch = tf.keras.applications.vgg19.preprocess_input(patch)
            x_test.append(patch)
        return x_test


def get_score(preds):
    m_score, f_score = 0, 0
    for prediction in preds:
        print('++++++++',prediction + prediction, prediction * 0.3)
        if prediction[0] > 0.5:
            f_score = f_score + 1
        else:
            m_score = m_score + 1
    print('male score : ', m_score)
    print('female score : ', f_score)
    print('classification : {0}'.format('MALE' if m_score > f_score else 'FEMALE'))
    print()
    return MALE if m_score > f_score else FEMALE


def random_crop(image):
    # crop image to random 400x400 scale image
    y = len(image)  # y is height
    x = len(image[0])  # x is width
    img = None
    while img is None:
        ranx = randint(0, x - 400)
        rany = randint(0, y - 400)
        div_im = image[rany:rany+400, ranx:ranx+400].copy()
        if count_not_white_pixels(div_im, 4500):
            img = div_im
    return img


os.chdir(r'F:\לימודים\פרויקט גמר\project')
model1 = load_model(model1_name)
model2 = load_model(model2_name)
model3 = load_model(model3_name)

n_files = len(glob.glob1(test_female_path, "*.jpg"))
f_predicted = 0
for image_name in os.listdir(test_female_path):
    if predict_image(test_female_path + '\\' + image_name, model1, model2, model3) == FEMALE:
        f_predicted += 1

m_predicted = 0
for image_name in os.listdir(test_male_path):
    if predict_image(test_male_path + '\\' + image_name, model1, model2, model3) == MALE:
        m_predicted += 1

print("summery :")
print('female predicted : {0} \\ {1}'.format( f_predicted, n_files))
print('male predicted : {0} \\ {1}'.format(m_predicted, n_files))

print('Accuracy : {0}%'.format(((f_predicted + m_predicted)/ (n_files * 2)) * 100))