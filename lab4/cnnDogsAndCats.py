from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time


def neural():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = 70

    data_transforms = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
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


    inputs, classes = next(iter(train_loader))
    print(inputs.shape)


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
            self.fc = nn.Linear(128 * 128 * 32, num_classes)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)  # флаттеринг
            out = self.fc(out)
            return out


    num_epochs = 3
    num_classes = 3

    # создаем экземпляр сети
    net = CnNet(num_classes).to(device)

    lossFn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    t = time.time()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # прямой проход
            outputs = net.forward(images)

            # вычисление значения функции потерь
            loss = lossFn(outputs, labels)

            # Обратный проход (вычисляем градиенты)
            optimizer.zero_grad()
            loss.backward()

            # делаем шаг оптимизации весов
            optimizer.step()

            # выводим немного диагностической информации
            if i % 100 == 0:
                print('Эпоха ' + str(epoch) + ' из ' + str(num_epochs) + ' Шаг ' +
                      str(i) + ' Ошибка: ', loss.item())

    print(time.time() - t)



    correct_predictions = 0
    num_test_samples = len(test_dataset)

    with torch.no_grad(): # о
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = net(images) # делаем предсказание по пакету
            _, pred_class = torch.max(pred.data, 1) # выбираем класс с максимальной оценкой
            correct_predictions += (pred_class == labels).sum().item()


    print('Точность модели: ' + str(100 * correct_predictions / num_test_samples) + '%')

    # Нашу модель можно сохранить в файл для дальнейшего использования
    torch.save(net.state_dict(), 'animals.ckpt')

if __name__ == '__main__':
    freeze_support()
    neural()