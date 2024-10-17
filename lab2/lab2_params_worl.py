import numpy as np
import pandas as pd

df = pd.read_csv('data.csv')

sampled = df.sample(140)
y = sampled.iloc[:, 4].values

# One-hot encoding для многоклассовой классификации
y = np.eye(3)[np.where(y == "Iris-setosa", 0, np.where(y == "Iris-versicolor", 1, 2))]

X = sampled.iloc[:, [0, 1, 2, 3]].values
X = np.concatenate([np.ones((len(X), 1)), X], axis=1)

def sigmoid(y):
    return 1 / (1 + np.exp(-y))

def derivative_sigmoid(y):
    return sigmoid(y) * (1 - sigmoid(y))

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum(axis=1, keepdims=True)

inputSize = X.shape[1]
hiddenSizes = 15
outputSize = 3

weights = [
    np.random.uniform(-2, 2, size=(inputSize, hiddenSizes)),
    np.random.uniform(-2, 2, size=(hiddenSizes, outputSize))
]

def feed_forward(x):
    input_ = x
    hidden_ = sigmoid(np.dot(input_, weights[0]))
    output_ = softmax(np.dot(hidden_, weights[1]))
    return [input_, hidden_, output_]

def backward(learning_rate, target, net_output, layers):
    err = (target - net_output)
    for i in range(len(layers) - 1, 0, -1):
        err_delta = err * derivative_sigmoid(layers[i])
        if err_delta.ndim == 1:
            err_delta = err_delta.reshape(-1, 1)
        layer_output = layers[i - 1]
        if layer_output.ndim == 1:
            layer_output = layer_output.reshape(1, -1)
        dw = np.dot(layer_output.T, err_delta)
        err = np.dot(err_delta, weights[i - 1].T)
        weights[i - 1] += learning_rate * dw

def train(x_values, target, learning_rate):
    output = feed_forward(x_values)
    backward(learning_rate, target, output[2], output)
    return None

def predict(x_values):
    return feed_forward(x_values)[-1]

iterations = 90
learning_rate = 0.02

for i in range(iterations):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    train(X_shuffled, y_shuffled, learning_rate)

    if i % 10 == 0:
        print("На итерации: " + str(i) + ' || ' + "Средняя ошибка: " + str(np.mean(np.square(y_shuffled - predict(X_shuffled)))))

# Тестирование на новой выборке
sampled = df.sample(135)
y_test = sampled.iloc[:, 4].values
y_test = np.eye(3)[np.where(y_test == "Iris-setosa", 0, np.where(y_test == "Iris-versicolor", 1, 2))]

X_test = sampled.iloc[:, [0, 1, 2, 3]].values
X_test = np.concatenate([np.ones((len(X_test), 1)), X_test], axis=1)

print("Тестируем")
pr = predict(X_test)

# Выбираем класс с наибольшей вероятностью для каждого примера
d = np.argmax(pr, axis=1)
y_tesе_r =np.argmax(y_test, axis=1)
# Считаем количество неверных предсказаний
count_nonzero = np.count_nonzero(abs(y_tesе_r - d))

print(f"Количество ошибок: {count_nonzero}")