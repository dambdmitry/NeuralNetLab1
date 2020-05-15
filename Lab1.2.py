import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.nn import *

#Загружаем данные в пандас.
data = pd.read_csv("apples_pears.csv")

#Истинный график
plt.figure(figsize = (10, 8))
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c = data['target'], cmap = 'rainbow')
plt.title('Яблоки и груши', fontsize = 15)
plt.xlabel('симметричность', fontsize = 14)
plt.ylabel('желтизна', fontsize= 14)
plt.show()

X = data.iloc[:,:2].values #Матрица объекты-признаки.
y = data['target'].values.reshape((-1, 1)) #Классы (столбец из 0 и 1)

num_features = X.shape[1] #Количество признаков

#Создаем один нейрон с функцией активации сигмоид.
neuron = torch.nn.Sequential(Linear(num_features, out_features=1), Sigmoid())

#Обернем данные в Torch
X = torch.autograd.Variable(torch.FloatTensor(X))
y = torch.autograd.Variable(torch.FloatTensor(y))

loss_fn = torch.nn.MSELoss(size_average = False) #Функция потерь.

learning_rate = 1 #Шаг градиентного спуска. (Был 0.01)

optimizer = torch.optim.Adam(neuron.parameters(), lr = learning_rate)

#Количество итераций было 500.
for t in range(2000):
    y_pred = neuron(X) #Выходные данные, предсказанные нейроном.

    loss = loss_fn(y_pred, y) #Функция потерь сверяет предсказанные данные с истинными.

    optimizer.zero_grad() #Обнуляем веса оптимизатора.

    loss.backward() #Вычисляем веса.

    optimizer.step() #Обновляем веса.

proba_pred = neuron(X)
y_pred = proba_pred > 0.5
y_pred = y_pred.data.numpy().reshape(-1) #Представляем предсказанные значения в виде столбца.

#График предсказания нейрона.
plt.figure(figsize=(10, 8))
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c = y_pred, cmap = 'rainbow')
plt.title('Предсказание нейрона "Яблоки и груши"', fontsize=15)
plt.xlabel('симметричность', fontsize = 14)
plt.ylabel('желтизна', fontsize = 14)
plt.show()