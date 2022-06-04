import cv2
import base64
import io
from random import randint
import numpy as np
import torch
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS
from flask import json
import flask
from PIL import Image
import torchvision
from AgeClassifier import AgeClassifier
from GenderClassification import GenderClassification

app = Flask(__name__)
CORS(app)
FEMALE = 0
MALE = 1
models = ["HHD", "KHATT", "QUWI"]
current_model = 'HHD'
age_model_path = "HHD_AGE.h5"
gender_model_path = "HHD_GENDER.h5"
 
scale = (299, 299)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
genderClassification = GenderClassification(modelPath='KHATT_GENDER.h5')
ageClassification = AgeClassifier(device=device, modelPath=age_model_path)

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
                                                                             
transform_quwi = torchvision.transforms.Compose([torchvision.transforms.Resize(448),
                                            torchvision.transforms.CenterCrop(448),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize([0.4856, 0.4994, 0.4324],
                                                                             [0.1817, 0.1811, 0.1927])])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def count_not_white_pixels(im, black_px_min):
    # count how many not white pixels in given image
    black = 0
    for i in range(im.shape[0]):  # traverses through height of the image
        for j in range(im.shape[1]):  # traverses through width of the image
            if im[i][j].all() < 200:
                black += 1
    return black > black_px_min  # return true if there is more black pixels than minimum black pixels


def image_segment(image):
    # divide image using 'slide window' to patches ,each patch is 400x400 scale
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


def rand_patch(image):
    # return a random 400x400 scale patch
    y = len(image)  # y is height
    x = len(image[0])  # x is width
    patch = None
    while patch is None:
        ranx = randint(0, x - 400)
        rany = randint(0, y - 400)
        im = image[rany:rany + 400, ranx:ranx + 400].copy()
        if count_not_white_pixels(im, 4500):
            patch = im
    return patch


def select_model(name):
    assert name in models
    if name in models:
        age_model = AgeClassifier(device=device, modelPath=name + '_AGE.h5')
        gender_model = GenderClassification(device=device, modelPath=name + '_GENDER.h5')
        return age_model, gender_model

def select_model_quwi(name):
    gender_model = GenderClassification(device=device, modelPath=name + '_GENDER.h5')
    return gender_model

def load_default_settings():
    return ageClassification, genderClassification


def predict_image(image, age_model, gender_model):
    patches = image_segment(image)  # extract patches from image
    if len(patches) % 2 == 0: patches.append(
        rand_patch(image))  # if number of patches is even, add 1 random patch to make it odd
    x_test = []
    for patch in patches:  # Process each patch to fit the model
        image = Image.fromarray(patch)
        img_tensor = transform(image).to(device).unsqueeze(0)
        x_test.append(img_tensor)

    patches_count = len(x_test)
    print('number of patches extracted from image : ', patches_count)
    gender_predictions = gender_model.predict(x_test)
    age_predections = age_model.predict(x_test)

    gender_score = [0, 0]
    for prediction in gender_predictions:
        gender_score[prediction[0]] = gender_score[prediction[0]] + 1

    age_score = [0, 0, 0, 0]
    for prediction in age_predections:
        age_score[prediction[0]] = age_score[prediction[0]] + 1

    classification = {
        "age": {
            "0": (age_score[0] / sum(age_score)) * 100,
            "1": (age_score[1] / sum(age_score)) * 100,
            "2": (age_score[2] / sum(age_score)) * 100,
            "3": (age_score[3] / sum(age_score)) * 100,
            "classification":int(np.argmax(np.array(age_score)))
        },
        "gender": {
            "female": (gender_score[FEMALE] / (gender_score[FEMALE] + gender_score[MALE])) * 100,
            "male": (gender_score[MALE] / (gender_score[FEMALE] + gender_score[MALE])) * 100,
            "classification": "MALE" if gender_score[MALE] > gender_score[FEMALE] else "FEMALE"
        },

    }

    return classification



def predict_image_quwi(image, gender_model):
    patches = image_segment(image)  # extract patches from image
    if len(patches) % 2 == 0: patches.append(
        rand_patch(image))  # if number of patches is even, add 1 random patch to make it odd
    x_test = []
    for patch in patches:  # Process each patch to fit the model
        image = Image.fromarray(patch)
        img_tensor = transform_quwi(image).to(device).unsqueeze(0)
        x_test.append(img_tensor)
    patches_count = len(x_test)
    print('number of patches extracted from image : ', patches_count)
    gender_predictions = gender_model.predict(x_test)

    gender_score = [0, 0]
    for prediction in gender_predictions:
        gender_score[prediction[0]] = gender_score[prediction[0]] + 1

    classification = {
        "gender": {
            "female": (gender_score[FEMALE] / (gender_score[FEMALE] + gender_score[MALE])) * 100,
            "male": (gender_score[MALE] / (gender_score[FEMALE] + gender_score[MALE])) * 100,
            "classification": "MALE" if gender_score[MALE] > gender_score[FEMALE] else "FEMALE"
        },
    }

    return classification


def QUWI():
    encoded = request.files['image'].read()
    model_name = request.form.get("model")
    #genderClassification = genderClassification
    genderClassification = select_model_quwi(model_name)
    decoded = base64.b64decode(encoded)  # decode data
    image_stream = io.BytesIO(decoded)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(encoded), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    print(image.shape)
    prediction = predict_image_quwi(image, genderClassification)
    response = jsonify(prediction)
    response.headers.set('Content-Type', 'custom')
    response.headers.set("Access-Control-Allow-Origin","*")
    return response



def Not_QUWI():
    encoded = request.files['image'].read()
    model_name = request.form.get("model")
    ageClassification, genderClassification = load_default_settings()
    if (model_name != current_model):
        print(model_name)
        ageClassification, genderClassification = select_model(model_name)
    decoded = base64.b64decode(encoded)  # decode data
    image_stream = io.BytesIO(decoded)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(encoded), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    print(image.shape)
    prediction = predict_image(image, ageClassification, genderClassification)
    response = jsonify(prediction)
    response.headers.set('Content-Type', 'custom')
    response.headers.set("Access-Control-Allow-Origin","*")
    
    return response

@app.route("/predict", methods=["POST"])
def predict():

    #print(request.data)
    #print(request.args)
    #message = request.get_json(force=True)  # Get image from user
    #encoded = request.files['image'].read()
    model_name = request.form.get("model")
    quwi_models = ['QUWI','QUWI_Ara','QUWI_Eng','QUWI_Ara_Eng']
    if model_name not in quwi_models:
         return Not_QUWI()
    else :
        return QUWI()
    '''
    ageClassification, genderClassification = load_default_settings()
    if (model_name != current_model):
        print(model_name)
        ageClassification, genderClassification = select_model(model_name)
    decoded = base64.b64decode(encoded)  # decode data
    image_stream = io.BytesIO(decoded)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(encoded), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    print(image.shape)
    prediction = predict_image(image, ageClassification, genderClassification)
    response = jsonify(prediction)
    response.headers.set('Content-Type', 'custom')
    response.headers.set("Access-Control-Allow-Origin","*")
    
    return response
    '''

if __name__ == '__main__':
    app.run()
