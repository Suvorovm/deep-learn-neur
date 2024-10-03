# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:24:56 2021

@author: AM4
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# загружаем и подготавляваем данные
df = pd.read_csv('data_err.csv')


df = df.iloc[np.random.permutation(len(df))]

trainIndexBorder = int(df.shape[0] * 0.8)

y = df.iloc[0:trainIndexBorder, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[0:trainIndexBorder, [0, 2]].values

y_test = df.iloc[trainIndexBorder:, 4].values
y_test = np.where(y_test == "Iris-setosa", 1, -1)
x_test = df.iloc[trainIndexBorder:, [0, 2]].values

inputSize = X.shape[1]  # количество входных сигналов равно количеству признаков задачи
hiddenSizes = 10  # задаем число нейронов скрытого (А) слоя
outputSize = 1 if len(y.shape) else y.shape[1]  # количество выходных сигналов равно количеству классов задачи

# создаем матрицу весов скрытого слоя
Win = np.zeros((1 + inputSize, hiddenSizes))
print(Win)
# пороги w0 задаем случайными числами
Win[0, :] = (np.random.randint(0, 3, size=(hiddenSizes)))
# остальные веса  задаем случайно -1, 0 или 1 
Win[1:, :] = (np.random.randint(-1, 2, size=(inputSize, hiddenSizes)))

# Wout = np.zeros((1+hiddenSizes,outputSize))

# случайно инициализируем веса выходного слоя
Wout = np.random.randint(0, 2, size=(1 + hiddenSizes, outputSize)).astype(np.float64)


# функция прямого прохода (предсказания) 
def predict(Xp):
    # выходы первого слоя = входные сигналы * веса первого слоя
    hidden_predict = np.where((np.dot(Xp, Win[1:, :]) + Win[0, :]) >= 0.0, 1, -1).astype(np.float64)
    # выходы второго слоя = выходы первого слоя * веса второго слоя
    out = np.where((np.dot(hidden_predict, Wout[1:, :]) + Wout[0, :]) >= 0.0, 1, -1).astype(np.float64)
    return out, hidden_predict


# обучение
# у перцептрона Розенблатта обучаются только веса выходного слоя 
# как и раньше обучаем подавая по одному примеру и корректируем веса в случае ошибки
def square_magnitude(vector):
    return sum(x * x for x in vector)


n_iter = 5
eta = 0.01
tolerance = 1e-6  # Для контроля изменения весов
last_w = np.zeros((Wout.shape[0], Wout.shape[1]))  # Начальное значение весов
accuracy = 0
weight_changes = []  # Список для хранения изменений весов

for i in range(n_iter):
    # Обновляем веса на основе данных
    for xi, target, j in zip(X, y, range(X.shape[0])):
        pr, hidden = predict(xi)
        Wout[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)
        Wout[0] += eta * (target - pr)


    current_weight_change = Wout - last_w


    if any(np.allclose(current_weight_change, wc, atol=tolerance) for wc in weight_changes):
        print(f"Weight changes repeated at iteration {i + 1}.")
        break


    weight_changes.append(np.copy(current_weight_change))


    last_w = np.copy(Wout)


    pr, hidden = predict(x_test)
    accuracy = 100 - ((np.count_nonzero(pr.reshape(-1) - y_test)) / y_test.shape[0]) * 100
    if accuracy >= 99.99:
        print(f"Desired accuracy reached: {accuracy:.2f}%")
        break


    weight_change_norm = np.linalg.norm(current_weight_change)
    if weight_change_norm < tolerance:
        print(f"Weights stopped changing after {i + 1} iterations.")
        break

# посчитаем сколько ошибок делаем на всей выборке
y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[:, [0, 2]].values
pr, hidden = predict(X)
print(f"Count elements {X.shape[0]}")
print((np.count_nonzero(pr.reshape(-1) - y)))

# далее оформляем все это в виде отдельного класса neural.py
