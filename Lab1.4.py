import  numpy as np
import matplotlib.pyplot as plt
import torch


N = 100
D = 2
K = 3

X = np.zeros((N * K, D))
y = np.zeros(N * K, dtype = 'uint8')

for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N) #N чисел от 0.0 до 1
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j

X = torch.autograd.Variable(torch.FloatTensor(X))
y = torch.autograd.Variable(torch.LongTensor(y.astype(np.int64)))

print(X.data.shape, y.data.shape)
# N - размер батча, D_in - размерность входа, H - размерность скрытых слоев, D_out - размерность выходного слоя
N, D_in, H, D_out  = 64, 2, 1000, 3

two_layer_net = torch.nn.Sequential(
    torch.nn.Linear(D_in, H), #Линейное преобразование ВХОД - СКРЫТЫЙ СЛОЙ
    torch.nn.ReLU(), #Функция активации
    torch.nn.Linear(H, D_out), #Линейное преобразование СКРЫТЫЙ СЛОЙ - ВЫХОД
)

loss_fn = torch.nn.CrossEntropyLoss(size_average = False) #Функция потерь
learning_rate = 9e-4
optimizer = torch.optim.SGD(two_layer_net.parameters(), lr = learning_rate)

#Обучение.
for t in range(250):
    y_pred = two_layer_net(X)

    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

#Преобразуем в NumPy для отрисовки.
X = X.data.numpy()
y = y.data.numpy()

#Отрисовка.
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
Z = two_layer_net(torch.autograd.Variable(grid_tensor))
Z = Z.data.numpy()
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(10, 8))

plt.contourf(xx, yy, Z, cmap=plt.cm.rainbow, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.rainbow)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('Игрушка дьявола', fontsize=15)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$y$', fontsize=14)
plt.show()