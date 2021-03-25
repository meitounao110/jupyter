import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import scipy.io as sio
from pathlib import Path
from torch.nn.functional import interpolate
from architectures.FPN1 import FPN_ResNet18
import os
import torch.nn as nn
from util.loss import Jacobi_layer, LaplaceLoss

# data1 = sio.loadmat(
#     "/mnt/layout_data/v0.3/data/one_point/train/train/Example0.mat"
# )
# u_true1 = data1["u"]  # 真实温度
# F1 = data1["f"]  # 布局
# fig1 = plt.figure(figsize=(10, 5))
# plt.subplot(121)
# im = plt.imshow(F1)
# plt.colorbar(im)
#
# plt.subplot(122)
# im = plt.imshow(u_true1)
# plt.colorbar(im)
# print(u_true1.max())
# print(u_true1.min())
# plt.show()

data = sio.loadmat(
    "/mnt/zhaoxiaoyu/data/layout_data/simple_component/one_point_dataset_200x200/train/Example1.mat"
)
u_true = data["u"]  # 真实温度
#F = data["list"]
load = data.get('list')[0]
# 需要修改
layout_map = np.zeros((200, 200))
mul = int(200 / 10)
for i in load:
    i = i - 1
    layout_map[(i % 10 * mul):((i % 10) * mul + mul), (i // 10 * mul):((i // 10 * mul) + mul)] = 10000 * np.ones(
        (mul, mul))
F = layout_map
# torch.set_default_dtype(torch.float32)
fig2 = plt.figure(figsize=(10, 5))
plt.subplot(121)
im = plt.imshow(F)
plt.colorbar(im)

plt.subplot(122)
im = plt.imshow(u_true)
plt.colorbar(im)
print(u_true.max())
print(u_true.min())
plt.show()

# er1 = u_true1.max() - u_true1.min()
# er = u_true.max() - u_true.min()
# a1 = (u_true1 - u_true1.min()) / er1
# a = (u_true - u_true.min()) / er
# b = abs(a1 * er + u_true.min() - u_true)
# mae = np.mean(b)
# print('mae', mae)

# laplaceLoss = Jacobi_layer()
# u_true_tensor = torch.from_numpy(u_true).type(torch.float64).unsqueeze(0).unsqueeze(0)
# print(u_true_tensor)

PATH = '/mnt/zhangyunyang1/pseudo_label-pytorch-master/experiments/model/semi_dw1_onepoint200_500l_7500ul_netc.pth'
model = FPN_ResNet18()
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['weight'])
model.eval()
F_tensor = torch.from_numpy(F).type(torch.float32).unsqueeze(0).unsqueeze(0)
u_true = torch.from_numpy(u_true).type(torch.float32).unsqueeze(0).unsqueeze(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
F_tensor = F_tensor.to(device)
u_true = u_true.to(device)
model = model.to(device)
loss = nn.L1Loss()
with torch.no_grad():
    u_pred = model(F_tensor / 10000)
    mae1 = loss(u_pred + 298, u_true)
a = torch.abs(u_pred - u_true + 298)
mae = torch.mean(a)
print(mae)
# u_pred_n = u_pred * 100 + 298
# x = laplaceLoss(F_tensor, (u_true_tensor - 298))
# u_pred_n = nn.L1Loss(u_true_tensor, x+298)
# u_pred_layout = u_pred_n.cpu().squeeze().numpy()
# x = x.cpu().squeeze().numpy()
# print(loss)
u_pred_n = u_pred.cpu().squeeze().numpy() + 298
# heat=heat.cpu().squeeze().numpy()
# f_pred=f_pred.squeeze().numpy()

fig2 = plt.figure(figsize=(10, 5))
plt.subplot(121)
im = plt.imshow(F)
plt.colorbar(im)

plt.subplot(122)
im = plt.imshow(u_pred_n)
plt.colorbar(im)
plt.show()
print(u_pred_n.max())
print(u_pred_n.min())

######################
root_dir = '/mnt/zhaoxiaoyu/data/layout_data/simple_component/one_point_dataset_200x200/train'
list_path = '/mnt/zhaoxiaoyu/data/layout_data/simple_component/dataset/200x200_val.txt'
mae_all = torch.zeros(2000)
files = []
with open(list_path, 'r') as rf:  # 打开list_path并读取其中内容
    for line in rf.readlines():
        data_path = line.strip()  # 移除字符串头尾的制定字符，此处为空格
        path = os.path.join(root_dir, data_path)  # path表示每个数据的路径
        files.append(path)
for idx in range(2000):
    data = sio.loadmat(files[idx])
    # data = sio.loadmat(
    #     f"../../data/test/Example{i}.mat"
    # )
    u_true = data["u"]
    load = data.get('list')[0]
    layout_map = np.zeros((200, 200))
    mul = int(200 / 10)
    for i in load:
        i = i - 1
        layout_map[(i % 10 * mul):((i % 10) * mul + mul), (i // 10 * mul):((i // 10 * mul) + mul)] = 10000 * np.ones(
            (mul, mul))
    F = layout_map
    F_tensor = torch.from_numpy(F).type(torch.float32).unsqueeze(0).unsqueeze(0) / 10000
    F_tensor = F_tensor.to(device)
    u_true = (torch.from_numpy(u_true).float().unsqueeze(0).unsqueeze(0))
    u_true = u_true.to(device)

    # u_pred = interpolate(F_tensor, scale_factor=4)
    with torch.no_grad():
        u_pred = model(F_tensor).cuda()
        maeloss = loss(u_pred + 298, u_true)

        # u_pre = (u_pred + 298).squeeze().cpu().numpy()
        # u_t = u_true.squeeze().cpu().numpy()
        # plt.imshow(u_pre)
        # plt.show()
        # plt.imshow(u_t)
        # plt.show()
        # plt.imshow(u_t - u_pre)
        # plt.colorbar()
        # plt.show()
        # print()
    # loss = nn.L1Loss()
    # maeloss = loss(u_pred, u_true.float())
    mae_all[idx] = maeloss.item()
mae = torch.mean(mae_all)
maestd = torch.std(mae_all)
print('mae_max', mae_all.max())
print('mae_min', mae_all.min())
print('mae', mae)
print('maestd', maestd)
####################
'''
# finetune100样本mae0.1551 方差0.0610 500样本mae 0.1118 方差0.0425 1000样本mae0.0815 方差0.0294
# 无监督mae0.2187 方差0.0969
# 半监督 100样本平均mae0.2777 方差0.1227 500样本平均mae0.2815 方差0.1216   1000样本平均mae0.1729 方差0.09
# 4000样本平均mae0.0307 方差0.0105 2000样本平均mae0.0654 方差0.0199 1000样本平均mae0.3198 方差0.0481 500样本平均mae0.439 方差0.0779 100样本平均mae0.7144 方差0.1018
x_data = ['1/80', '1/16', '1/8', '1/4', '1/2']
y_data = [0.7144, 0.439, 0.3198, 0.0654, 0.0307]
# x_data = ['1/80', '1/16', '1/8']
# y_data = [0.7144, 0.439, 0.3198]
# y_data1 = [0.2777, 0.2815, 0.1729]
# y_data2 = [0.1551, 0.1118, 0.0815]
# y_data2 = [52000, 54200, 51500, 58300, 56800, 59500, 62700]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
ln1, = plt.plot(x_data, y_data, color='red', linewidth=2.0, )
# ln2, = plt.plot(x_data, y_data1, color='blue', linewidth=2.0, linestyle='--')
# ln3, = plt.plot(x_data, y_data2, color='black', linewidth=2.0, linestyle='-.')
my_font = fm.FontProperties(fname="/mnt/simsun.ttf")

plt.title("mae指标随样本量变化", fontsize=12, fontproperties=my_font)  # 设置标题及字体
plt.legend(handles=[ln1], labels=['全监督'], prop=my_font)
#plt.legend(handles=[ln1, ln2, ln3], labels=['全监督', '自监督', 'fintune'], prop=my_font)
# plt.legend(handles=[ln2], labels=['半监督'], prop=my_font)

plt.xlabel("有标记样本量", fontsize=12, fontproperties=my_font)
plt.ylabel("mae", fontsize=12)

plt.show()
'''