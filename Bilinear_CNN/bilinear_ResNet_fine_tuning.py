import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
from torchvision import datasets
import os
import bilinear_resnet
import CUB_200

data_dir = '../HHD_gender'
base_lr = 0.001
batch_size = 24
num_epochs = 50
weight_decay = 1e-5
num_classes = 2
cub200_path = 'data'
save_model_path = 'model_saved/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train():
    model = bilinear_resnet.BCNN(num_classes, pretrained=False).to(device)

    model_d = model.state_dict()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay)
    #optimizer = optim.Adam(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    train_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(448),
                                                      torchvision.transforms.CenterCrop(448),
                                                      torchvision.transforms.RandomHorizontalFlip(),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize([0.4856, 0.4994, 0.4324],
                                                                                       [0.1817, 0.1811, 0.1927])])
    test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(448),
                                                     torchvision.transforms.CenterCrop(448),
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize([0.4856, 0.4994, 0.4324],
                                                                                      [0.1817, 0.1811, 0.1927])])

    # train_data = CUB_200.CUB_200(cub200_path, train=True, transform=train_transform)
    # test_data = CUB_200.CUB_200(cub200_path, train=False, transform=test_transform)
    train_data = datasets.ImageFolder(data_dir + f'/train', train_transform)
    test_data = datasets.ImageFolder(data_dir + f'/val', test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    print(len(train_data))
    print('Start fine-tuning...')
    best_acc = 0.
    best_epoch = None
    end_patient = 0
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        epoch_loss = 0.
        loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            _, prediction = torch.max(outputs.data, 1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)

            print(f'\rEpoch {epoch + 1}: Iter {i + 1}, Train loss {loss}')
        train_acc = 100 * correct / total
        print('Testing on test dataset...')
        test_acc = test_accuracy(model, test_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss} Train_Acc: {train_acc}  Test_Acc: {test_acc}')
        if test_acc > best_acc:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,

            }, "bilinear_resnet.h5")
        scheduler.step(test_acc)


def test_accuracy(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images)

            _, prediction = torch.max(outputs.data, 1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)
        model.train()
        return 100 * correct / total


def main():
    train()


if __name__ == '__main__':
    main()
