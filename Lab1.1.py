import torch

#Задание 1.
print("Первое задание")
a = torch.FloatTensor(3, 4)
b = torch.FloatTensor(12)
c = b.view(2, 2, 3)
print(a)
#чуть не понял задание, что понимать под "столбцом" матрицы, поэтому тут 2 варианта.
print(a[1])
print(a[:, 1])
print()
#Задание 2.
a = torch.FloatTensor(5, 2).random_(1, 10)
b = torch.FloatTensor(1, 10).random_(1, 10)
c = b.view(5, 2)

print("a ",a)
print("c ", c)

print("Сложение ", a.add(c))
print("Вычитание ",a.sub(c))
print("Умножение ",a.mul(c))
print("Деление", a.div(c))

#Задание 3.
a = torch.FloatTensor(100, 780, 780, 3).random_(1, 100)

print(a[0].mean())
print(a[3].mean())

