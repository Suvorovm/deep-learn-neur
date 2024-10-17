# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:29:37 2021

@author: AM4
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data.csv')


sampled = df.sample(135)
y = sampled.iloc[:, 4].values

y = np.where(y == "Iris-setosa", 0,
    np.where(y == "Iris-versicolor", 1, 2)).reshape(-1, 1)

X = sampled.iloc[:, [0,1,2,3]].values

# добавим фиктивный признак для удобства матричных вычслений
X = np.concatenate([np.ones((len(X), 1)), X], axis=1)


# зададим функцию активации - сигмоида
def sigmoid(y):
    return 1 / (1 + np.exp(-y))


# нам понадобится производная от сигмоиды при вычислении градиента
def derivative_sigmoid(y):
    return sigmoid(y) * (1 - sigmoid(y))


# инициализируем нейронную сеть
inputSize = X.shape[1]  # количество входных сигналов равно количеству признаков задачи
hiddenSizes = 50  # задаем число нейронов скрытого слоя
outputSize = 3  # количество выходных сигналов равно количеству классов задачи

# веса инициализируем случайными числами, но теперь будем хранить их списком
weights = [
    np.random.uniform(-2, 2, size=(inputSize, hiddenSizes)),  # веса скрытого слоя
    np.random.uniform(-2, 2, size=(hiddenSizes, outputSize))  # веса выходного слоя
]

# Функция активации - ReLU для скрытого слоя
def relu(y):
    return np.maximum(0, y)

# Производная ReLU
def derivative_relu(y):
    return np.where(y > 0, 1, 0)

# Softmax для выходного слоя
def softmax(y):
    exp_values = np.exp(y - np.max(y, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Прямой проход
def feed_forward(x):
    input_ = x
    hidden_ = relu(np.dot(input_, weights[0]))
    output_ = softmax(np.dot(hidden_, weights[1]))  # Используем softmax для выхода
    return [input_, hidden_, output_]

# Backpropagation
def backward(learning_rate, target, net_output, layers):
    # Ошибка: используем cross-entropy loss
    err = net_output - target

    for i in range(len(layers) - 1, 0, -1):
        if i == len(layers) - 1:
            err_delta = err  # Производная от softmax + cross-entropy
        else:
            err_delta = err * derivative_relu(layers[i])

        err = np.dot(err_delta, weights[i - 1].T)
        dw = np.dot(layers[i - 1].T, err_delta)
        weights[i - 1] -= learning_rate * dw

# One-hot encoding для целевых меток
def one_hot_encoding(y, num_classes):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y.flatten()] = 1
    return one_hot

# Применение

# Обучаем сеть с новыми параметрами


# функция обучения чередует прямой и обратный проход
def train(x_values, target, learning_rate):
    output = feed_forward(x_values)
    backward(learning_rate, target, output[2], output)
    return None


# функция предсказания возвращает только выход последнего слоя
def predict(x_values):
    return feed_forward(x_values)[-1]


# задаем параметры обучения
iterations = 90
learning_rate = 0.01

y_one_hot = one_hot_encoding(y, outputSize)

# обучаем сеть (фактически сеть это вектор весов weights)
for i in range(iterations):
    train(X, y, learning_rate)

    if i % 10 == 0:
        print("На итерации: " + str(i) + ' || ' + "Средняя ошибка: " + str(np.mean(np.square(y - predict(X)))))


sampled = df.sample(135)
y = sampled.iloc[:, 4].values

y = np.where(y == "Iris-setosa", 0,
    np.where(y == "Iris-versicolor", 1, 2)).reshape(-1, 1)


X = sampled.iloc[:, [0,1,2,3]].values

# добавим фиктивный признак для удобства матричных вычслений
X = np.concatenate([np.ones((len(X), 1)), X], axis=1)

print("Тестируем")
pr = predict(X)


# Выбираем класс с наибольшей вероятностью для каждого примера
d = np.argmax(pr, axis=1)

# Считаем количество неверных предсказаний
count_nonzero = np.count_nonzero(abs(y.reshape(-1) - d))

print(f"Количество ошибок: {count_nonzero}")


## Вероятность Н?
