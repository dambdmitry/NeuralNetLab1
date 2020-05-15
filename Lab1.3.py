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

plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:,1], c = y, s = 40, cmap = "rainbow")
plt.title("Игрушка дбявола", fontsize = 15)
plt.xlabel('$x$', fontsize=14)
plt.ylabel("$y$", fontsize=14)
plt.show()

X = torch.autograd.Variable(torch.FloatTensor(X))
y = torch.autograd.Variable(torch.LongTensor(y.astype(np.int64)))

print(X.data.shape, y.data.shape)

N, D_in, D_out = 64, 2, 3

neuron = torch.nn.Sequential(torch.nn.Linear(D_in, D_out),)

loss_fn = torch.nn.NLLLoss(size_average = False)

learning_rate = 9e-4

optimizer = torch.optim.SGD(neuron.parameters(), lr = learning_rate)

for t in range(250):
    y_pred = neuron(X)

    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

X = X.data.numpy()
y = y.data.numpy()

h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

Z = neuron(torch.autograd.Variable(grid_tensor))
Z = Z.data.numpy()
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))

plt.contourf(xx, yy, Z, cmap = "rainbow", alpha = 0.3)
plt.scatter(X[:, 0], X[:, 1], c = y, s = 40, cmap = "rainbow")

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.title("Игрушка дьявола", fontsize = 15)
plt.xlabel("$x$", fontsize = 14)
plt.ylabel("$y$", fontsize = 14)
plt.show()