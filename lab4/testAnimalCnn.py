from multiprocessing import freeze_support
from random import random
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = 70  ## ВИЗУАЛИЗАЦИЯ ??

    data_transforms = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], ## ПОЧЕМУ ??
                             std=[0.229, 0.224, 0.225])
    ])

    train_data = torchvision.datasets.ImageFolder(root='./animals/train',
                                                     transform=data_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=True, num_workers=8)


    print(f"Количество классов {train_data.classes} Количестов примеров  {len(train_data.samples)}")
    print(device)

    class_names = train_data.classes

    test_dataset = torchvision.datasets.ImageFolder(root='./animals/val',
                                                 transform=data_transforms)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=8)

    print(f"Количество классов {test_dataset.classes} Количестов примеров  {len(test_dataset.samples)}")
    class CnNet(nn.Module):
        def __init__(self, num_classes=3):
            nn.Module.__init__(self)
            self.layer1 = nn.Sequential(
                # первый сверточный слой с ReLU активацией и maxpooling-ом
                nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            # второй сверточный слой с ReLU активацией и maxpooling-ом
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            # классификационный слой
            self.fc = nn.Linear(128 * 128 * 32, num_classes) ## ВИЗУАЛИЗАЦИЯ ?? Fc ??

        def forward(self, x): ## ВИЗУАЛИЗАЦИЯ ??
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)  # флаттеринг
            out = self.fc(out)
            return out




    net = CnNet().to(device)  # Инициализация модели
    net.eval()  # Перевод модели в режим оценки
    correct_predictions = 0
    num_test_samples = len(test_dataset)
    net.load_state_dict(torch.load('animals.ckpt'))

    all_predictions = []
    all_labels = []
    sample_labels = []

    sample_images = []
    sample_predictions = []

    with torch.no_grad(): # о
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = net(images) # делаем предсказание по пакету
            _, pred_class = torch.max(pred.data, 1) # выбираем класс с максимальной оценкой
            all_predictions.extend(pred_class.cpu().numpy())  # Перевод на CPU и преобразование в numpy
            all_labels.extend(labels.cpu().numpy())  # То же самое для меток
            correct_predictions += (pred_class == labels).sum().item()

            sample_images.extend(images.cpu().numpy())  # Сохраняем изображения на CPU
            sample_predictions.extend(pred_class.cpu().numpy())  # Сохраняем предсказания
            sample_labels.extend(labels.cpu().numpy())

    print('Точность модели: ' + str(100 * correct_predictions / num_test_samples) + '%')
    report = classification_report(all_labels, all_predictions, target_names=test_dataset.classes)

    print(report)

    random_indices = random.sample(range(len(sample_images)), 10)

    # Отображаем 10 случайных изображений и их предсказания
    plt.figure(figsize=(15, 8))
    for i, idx in enumerate(random_indices):
        plt.subplot(2, 5, i + 1)  # Размещаем 10 изображений в сетке 2x5
        img = np.transpose(sample_images[idx], (1, 2, 0))  # Преобразуем изображение в формат HxWxC

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        # Отображаем изображение и добавляем текст с предсказанным и истинным классом
        plt.imshow(img)
        true_class = sample_labels[idx]
        predicted_class = sample_predictions[idx]
        plt.title(f'True: {test_dataset.classes[true_class]}\nPred: {test_dataset.classes[predicted_class]}')
        plt.axis('off')  # Скрываем оси
    plt.show()



if __name__ == '__main__':
    freeze_support()
    test()

