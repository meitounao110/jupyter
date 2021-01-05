import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io as sio
from pathlib import Path
from torch.nn.functional import interpolate
from architectures.fpn import Fpn_n
import os
import torch.nn as nn

data = sio.loadmat(
    "/mnt/layout_data/v0.3/data/one_point/test/0/test/Example10001.mat"
)
u_true = data["u"]  # 真实温度
F = data["f"]  # 布局

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

PATH = '/mnt/zhangyunyang1/pseudo_label-pytorch-master/experiments/model/onepoints_400_1600_FPN_model.pth'
model = Fpn_n()
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['weight'])
model.eval()

F_tensor = (torch.from_numpy(F.astype(float)).float().unsqueeze(0).unsqueeze(0)) / 20000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
F_tensor = F_tensor.to(device)
model = model.to(device)
with torch.no_grad():
    u_pred = model(F_tensor)
u_pred = interpolate(u_pred, size=(200, 200),
                     mode='bilinear',
                     align_corners=True) * 100 + 298
u_pred_n = u_pred.cpu().squeeze().numpy()

fig2 = plt.figure(figsize=(10, 5))
plt.subplot(121)
im = plt.imshow(F)
plt.colorbar(im)

plt.subplot(122)
im = plt.imshow(u_pred_n)
plt.colorbar(im)
# fig.savefig('./predict.png', dpi=100)
plt.show()
print(u_pred_n.max())
print(u_pred_n.min())
a = torch.abs(torch.Tensor(u_pred_n - u_true))
mae = torch.mean(a)
print('mae', mae)

'''
root_dir = '/mnt/layout_data/v0.3/data/one_point/test/0/test'
list_path = '/mnt/layout_data/v0.3/data/one_point/test/test_0.txt'
mae_all = torch.Tensor(np.zeros(10000))
files = []
with open(list_path, 'r') as rf:  # 打开list_path并读取其中内容
    for line in rf.readlines():
        data_path = line.strip()  # 移除字符串头尾的制定字符，此处为空格
        path = os.path.join(root_dir, data_path)  # path表示每个数据的路径
        files.append(path)
for i in range(10000):
    data = sio.loadmat(files[i])
    # data = sio.loadmat(
    #     f"../../data/test/Example{i}.mat"
    # )
    u_true = data["u"]
    F = data["f"]
    F = F.astype(float)
    F_tensor = (torch.from_numpy(F).float().unsqueeze(0).unsqueeze(0)) / 20000
    F_tensor = F_tensor.to(device)
    u_true = (torch.from_numpy(u_true).float().unsqueeze(0).unsqueeze(0))
    u_true = u_true.to(device)
    u_std = 100
    u_mean = 298

    # u_pred = interpolate(F_tensor, scale_factor=4)
    with torch.no_grad():
        u_pred = model(F_tensor).cuda()
    u_pred = interpolate(u_pred, size=(200, 200), mode='bilinear',
                         align_corners=True) * u_std + u_mean
    # u_pred = interpolate(u_pred, scale_factor=4)
    loss = nn.L1Loss()
    maeloss = loss(u_pred, u_true.float())
    mae_all[i] = maeloss.item()
    # u_pred_n = u_pred.squeeze()

    # mae_all[i] = torch.mean(torch.abs(torch.Tensor(u_pred_n - u_true)))

    # a = torch.abs(u_pred_n - u_true).cuda()
    # mae_all[i] = torch.mean(a).cuda()
mae = torch.mean(mae_all).cuda()
print('mae_max', mae_all.max())
print('mae_min', mae_all.min())
print('mae', mae)
'''
