import torch
from torchvision.transforms import transforms
from PIL import Image
from cnn_main import CNNet
from pathlib import Path

model = CNNet(5)
checkpoint = torch.load(Path("/home/genderclassification/fineGrained/Bilinear_CNN/bilinear_resnet.h5")
model.load_state_dict(checkpoint)
'''
trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])
'''
image = Image.open(Path("/home/genderclassification/fineGrained/HHD_gender/val/female/BRN3C2AF4AEB56C_0000000076.jpg"))

input = trans(image)

input = input.view(1, 3, 32,32)

output = model(input)

prediction = int(torch.max(output.data, 1)[1].numpy())
print(prediction)