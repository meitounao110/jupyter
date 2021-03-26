import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import torch

data = sio.loadmat(
    "/mnt/zhangyunyang1/pseudo_label-pytorch-master/Example.mat"
)
u_true = data["U"]  # 真实温度
load = data.get('F')
# 需要修改
# layout_map = np.zeros((200, 200))
# mul = int(200 / 10)
# for i in load:
#     i = i - 1
#     layout_map[(i % 10 * mul):((i % 10) * mul + mul), (i // 10 * mul):((i // 10 * mul) + mul)] = 10000 * np.ones(
#         (mul, mul))
F = load
fig1 = plt.figure(figsize=(10, 5))
plt.subplot(121)
im = plt.imshow(F)
plt.colorbar(im)

plt.subplot(122)
im = plt.imshow(u_true)
plt.colorbar(im)
print(u_true.max())
print(u_true.min())
plt.show()

data = sio.loadmat(
    "/mnt/zhangyunyang1/pseudo_label-pytorch-master/Example_with01.mat"
)
u_true1 = data["U"]  # 真实温度
load1 = data.get('F')
# 需要修改
# layout_map = np.zeros((200, 200))
# mul = int(200 / 10)
# for i in load:
#     i = i - 1
#     layout_map[(i % 10 * mul):((i % 10) * mul + mul), (i // 10 * mul):((i // 10 * mul) + mul)] = 10000 * np.ones(
#         (mul, mul))
F1 = load1
fig2 = plt.figure(figsize=(10, 5))
plt.subplot(121)
im = plt.imshow(F1)
plt.colorbar(im)

plt.subplot(122)
im = plt.imshow(u_true1)
plt.colorbar(im)
print(u_true1.max())
print(u_true1.min())
plt.show()

a = (u_true - 298) / (u_true.max() - u_true.min())
a1 = (u_true1 - 298) / (u_true1.max() - u_true1.min())
mae = np.mean(abs(a - a1))
print(mae)

mae1=np.mean(abs(u_true - u_true1))
print(mae1)
