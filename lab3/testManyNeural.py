import numpy as np
import pandas as pd
import torch
from torch import nn

from lab3.neural import MLPptorch

df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, 0).reshape(-1,1)
X = df.iloc[0:100, 0:3].values


x_test = df.iloc[100:-1, 0:3].values
y_test = df.iloc[100:-1, 4].values
y_test = np.where(y == "Iris-setosa", 1, 0).reshape(-1,1)


inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи
hiddenSizes = [10, 12 , 13]
outputSize = 1 if len(y.shape) else y.shape[1]

model = MLPptorch(inputSize, hiddenSizes, outputSize, nn.Sigmoid())

num_iter = 1000

print("sigmoid")
lossFn = nn.MSELoss()
optimizer  = torch.optim.SGD(model.parameters(), lr=0.01)


for i in range(0, num_iter):
    pred = model.forward(torch.from_numpy(X.astype(np.float32)))
    loss = lossFn(pred, torch.from_numpy(y.astype(np.float32)))
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print('Ошибка на ' + str(i + 1) + ' итерации: ', loss.item())



pred = model.forward(torch.from_numpy(x_test.astype(np.float32)))
loss = lossFn(pred, torch.from_numpy(y_test.astype(np.float32)))
print(f"\nОшибка для тестовой выборки {loss.item()}\n")

print(f"\nПараметры для сегмоиды \n")
for name, param in model.named_parameters():
    print(name, param)




print("\n\n Экспиремент для RELU \n\n")


hiddenSizes = [5, 7 , 8]
model = MLPptorch(inputSize, hiddenSizes, outputSize, nn.ReLU())
lossFn = nn.MSELoss()
optimizer  = torch.optim.SGD(model.parameters(), lr=0.00001)


for i in range(0, 1000):
    pred = model.forward(torch.from_numpy(X.astype(np.float32)))
    loss = lossFn(pred, torch.from_numpy(y.astype(np.float32)))
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print('Ошибка на ' + str(i + 1) + ' итерации: ', loss.item())

print("\n\n проверка")

pred = model.forward(torch.from_numpy(x_test.astype(np.float32)))
loss = lossFn(pred, torch.from_numpy(y_test.astype(np.float32)))
print(f"\nОшибка для тестовой выборки {loss.item()}\n")

print(f"\nПараметры для сегмоиды \n")
for name, param in model.named_parameters():
    print(name, param)
