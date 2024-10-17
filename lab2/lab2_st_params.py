# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:29:37 2021

@author: AM4
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data.csv')


sampled = df.sample(140)
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
hiddenSizes = 15  # задаем число нейронов скрытого слоя
outputSize = 3  # количество выходных сигналов равно количеству классов задачи

# веса инициализируем случайными числами, но теперь будем хранить их списком
weights = [
    np.random.uniform(-2, 2, size=(inputSize, hiddenSizes)),  # веса скрытого слоя
    np.random.uniform(-2, 2, size=(hiddenSizes, outputSize))  # веса выходного слоя
]


# прямой проход
def feed_forward(x):
    input_ = x  # входные сигналы
    hidden_ = sigmoid(np.dot(input_, weights[0]))  # выход скрытого слоя = сигмоида(входные сигналы*веса скрытого слоя)
    output_ = sigmoid(
        np.dot(hidden_, weights[1]))  # выход сети (последнего слоя) = сигмоида(выход скрытого слоя*веса выходного слоя)

    # возвращаем все выходы, они нам понадобятся при обратном проходе
    return [input_, hidden_, output_]


# backprop собственной персоной
# на вход принимает скорость обучения, реальные ответы, предсказанные сетью ответы и выходы всех слоев после прямого прохода
def backward(learning_rate, target, net_output, layers):
    # считаем производную ошибки сети
    err = (target - net_output)

    # прогоняем производную ошибки обратно ко входу, считая градиенты и корректируя веса
    # для этого используем chain rule
    # цикл перебирает слои от последнего к первому
    for i in range(len(layers) - 1, 0, -1):
        # градиент слоя = ошибка слоя * производную функции активации * на входные сигналы слоя

        # ошибка слоя * производную функции активации
        err_delta = err * derivative_sigmoid(layers[i])  ## Why  layer[i]

        # пробрасываем ошибку на предыдущий слой
        err = np.dot(err_delta, weights[i - 1].T)

        # ошибка слоя * производную функции активации * на входные сигналы слоя
        dw = np.dot(layers[i - 1].T, err_delta)

        # обновляем веса слоя
        weights[i - 1] += learning_rate * dw


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
learning_rate = 0.02

# обучаем сеть (фактически сеть это вектор весов weights)
for i in range(iterations):

    # создаем массив индексов
    indices = np.arange(len(X))

    # перемешиваем индексы
    np.random.shuffle(indices)

    # перемешиваем X и y по индексу
    X_shuffled = X[indices]
    y_shuffled = y[indices]


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
count_nonzero = np.count_nonzero(abs(y.reshape(-1) - np.round(d)))

print(f"Количество ошибок: {count_nonzero}")


## Вероятность Н?
