from torchvision import datasets
import torchvision
from GenderClassification import GenderClassification
from AgeClassifier import AgeClassifier
import torch
import cv2
from PIL import Image
import numpy as np


def to_negativ(x: Image):
    im_np = np.asarray(x)
    im_np = 255 - im_np
    im_pil = Image.fromarray(im_np)
    return im_pil








transform = torchvision.transforms.Compose([torchvision.transforms.Resize(448),
                                            torchvision.transforms.CenterCrop(448),
                                            #torchvision.transforms.Lambda(to_negativ),
                                            #torchvision.transforms.RandomRotation(30),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize([0.4856, 0.4994, 0.4324],
                                                                             [0.1817, 0.1811, 0.1927])])
batch_size = 32

train_DataDir = '/home/languagedetection/Datasets/gender-age/quwi-1/segmented/arabic/train/'
train_eng_Datader = '/home/languagedetection/Datasets/gender-age/quwi-1/segmented/english/train/'
train_data = datasets.ImageFolder(train_DataDir, transform)
train_data2 = datasets.ImageFolder(train_eng_Datader, transform)
train_data_loader = torch.utils.data.ConcatDataset([train_data, train_data2])
train_loader = torch.utils.data.DataLoader(dataset=train_data_loader, batch_size=batch_size, shuffle=True)


test_DataDir = '/home/languagedetection/Datasets/gender-age/quwi-1/segmented/english/test/'
test_DataDir = '/home/languagedetection/Datasets/gender-age/quwi-1/segmented/arabic/test/'
test_data = datasets.ImageFolder(test_DataDir, transform)
test_data2 = datasets.ImageFolder(test_DataDir, transform)
test_data_loader = torch.utils.data.ConcatDataset([test_data, test_data2])
test_loader = torch.utils.data.DataLoader(dataset=test_data_loader, batch_size=batch_size, shuffle=False)


val_DataDir = '/home/languagedetection/Datasets/gender-age/quwi-1/segmented/arabic/valid/'
val1_DataDir = '/home/languagedetection/Datasets/gender-age/quwi-1/segmented/english/valid/'
valid_data2 = datasets.ImageFolder(val1_DataDir, transform)
val_data = datasets.ImageFolder(val_DataDir, transform)
val_data_loader = torch.utils.data.ConcatDataset([val_data, valid_data2])
val_loader = torch.utils.data.DataLoader(dataset=val_data_loader, batch_size=batch_size, shuffle=False)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
ageClassification = GenderClassification(device=device)
#ageClassification = GenderClassification(device=device, modelPath='QUWI_Full_Text_Arabic_Script_dependent-74.66666666666667.h5')
ageClassification.train(train_loader, val_loader, 20, 0.1, 1e-8, 'QUWI_Ara_Eng_Patched_SGD')
print(ageClassification.test_accuracy(test_loader))
