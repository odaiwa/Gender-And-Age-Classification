from Bilinear_CNN import bilinear_resnet
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


class AgeClassifier:
    def __init__(self, modelPath=None, device='cpu'):
        self.device = device
        self.model = bilinear_resnet.BCNN(4, pretrained=False).to(device)
        self.criterion = nn.CrossEntropyLoss()
        if modelPath:
            checkpoint = torch.load(modelPath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def train(self, trainData, testData, num_epochs, learningRate, weight_decay, modelName):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learningRate, momentum=0.9, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3,
                                                               verbose=True)
        print('Start fine-tuning...')
        best_acc = 0.
        self.model.train()
        for epoch in range(num_epochs):
            correct = 0
            total = 0
            epoch_loss = 0.
            loss = 0
            for i, (images, labels) in enumerate(trainData):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs, _ = self.model(images)
                loss = self.criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss
                _, prediction = torch.max(outputs.data, 1)
                correct += (prediction == labels).sum().item()
                total += labels.size(0)

                print(f'Epoch {epoch + 1}: Iter {i + 1}, Train loss {loss}', end='\r')
            train_acc = 100 * correct / total
            print('Testing on test dataset...', end='\r')
            self.model.eval()
            test_acc,test_loss = self.test_accuracy(testData)
            self.model.train()
            print(f'Epoch [{epoch + 1}/{num_epochs}]: train Loss: {loss} Train_Acc: {train_acc}  test Loss: {test_loss} Test_Acc: {test_acc}')
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,

                }, f'{modelName}-{best_acc}.h5')
            scheduler.step(test_acc)

    def test_accuracy(self, data):
        with torch.no_grad():
            correct = 0
            total = 0
            loss=0
            for images, labels in data:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs, _ = self.model(images)
                _, prediction = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)
                correct += (prediction == labels).sum().item()
                total += labels.size(0)
            return 100 * correct / total , loss

    def predict(self, images:list):
        self.model.eval()
        predictions=[]
        for image in images:
            prediction,_ = self.model(image)
            _, prediction = torch.max(prediction,1)
            predictions.append(prediction.cpu().numpy())
        return predictions
