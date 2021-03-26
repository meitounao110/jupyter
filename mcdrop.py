# %% md

# 部署模型

# %%

import sys

sys.path.append('/home/zhengxiaohu/xx/2020.8.31')

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import np_transforms as transforms
from layout import LayoutDataset
from torch.utils.data import DataLoader
from fpn_head import FPN
from mat2pic import GeneralDataset, TestDataset, trans_separate
from scheduler import WarmupMultiStepLR
from tqdm import tqdm, trange
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

TRAIN_EPOCHS = 1000
BATCH_SIZE = 32
TEST_BATCH_SIZE = 1
TEST_SAMPLES = 500

# %%

train_dataset = GeneralDataset(trans_separate, resize_shape=(200, 200))
test_dataset = TestDataset(trans_separate, resize_shape=(200, 200))

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

TRAIN_SIZE = len(train_dataloader.dataset)
TEST_SIZE = len(test_dataloader.dataset)
NUM_BATCHES = len(train_dataloader)
NUM_TEST_BATCHES = len(test_dataloader)

# %%

# def forward(self, x, training_=True, training_mc=True):
#     if training_:
#         x = self.backbone(x)
#         _, c2, c3, c4, c5 = x

#         p5 = self.conv1(c5)
#         p4 = self.p4([p5, c4])
#         p3 = self.p3([p4, c3])
#         p2 = self.p2([p3, c2])

#         s5 = self.s5(p5, sizes = [c4.size()[-2:], c3.size()[-2:], c2.size()[-2:]])
#         s4 = self.s4(p4, sizes = [c3.size()[-2:], c2.size()[-2:]])
#         s3 = self.s3(p3, sizes = [c2.size()[-2:]])
#         s2 = self.s2(p2, sizes = [c2.size()[-2:]])

#         x = s5 + s4 + s3 + s2

#     if training_mc:

#         x = self.dropout(x)
#         x = self.final_conv(x)

#         if self.final_upsampling is not None and self.final_upsampling > 1:
#             x = F.interpolate(x, scale_factor=self.final_upsampling, mode='bilinear', align_corners=True)
#     return x

net = FPN().to(DEVICE)
net.load_state_dict(torch.load('./mcd_fpn.pkl'))
# net.load_state_dict(torch.load('./mcd_fpn_last.pkl'))

# %%

loss_pic = []


def Train(net, optimizer, epoch, TRAIN_EPOCHS):
    global loss_pic

    net.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_dataloader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        pre = net(data, training_=True, training_mc=True)
        loss = F.l1_loss(pre, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    net.eval()
    avg = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_dataloader)):
            data, target = data.to(DEVICE), target.to(DEVICE)
            mdata = scio.loadmat('/home/zhengxiaohu/xx/Data/data_200*200_v2/test/' + str(batch_idx + 1) + '.mat')

            output_ = net(data, training_=True, training_mc=True)
            test_loss = torch.mean(torch.abs(output_.cpu() * 100 + 260 - mdata['u']))
            avg += test_loss.item()
        avg = avg / len(test_dataloader)

    if epoch > 0 and avg < min(loss_pic):
        torch.save(net.state_dict(), './mcd_fpn.pkl')
    loss_pic.append(avg)

    print("Epoch: [{}/{}], Test Loss: {:.4f}".format(epoch + 1, TRAIN_EPOCHS, avg))


# %%

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = WarmupMultiStepLR(optimizer,
                              milestones=[],
                              warmup_iters=len(train_dataloader))
for epoch in range(TRAIN_EPOCHS):
    Train(net, optimizer, epoch, TRAIN_EPOCHS)

# %%

torch.save(net.state_dict(), './mcd_fpn_last.pkl')

# %%

# np.save('loss_pic.npy',np.array(loss_pic))
plt.plot(range(1, 1001), loss_pic)
plt.ylim(0, 2)
plt.show()

# %% md

# 测试加可视化

# %%

test_loss = 0
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(tqdm(test_dataloader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE, 1, 200, 200).to(DEVICE)
        #         outputs_initial = net(data, training_=True, training_mc=False)
        for i in range(TEST_SAMPLES):
            outputs[i] = net(data, training_=True, training_mc=True)
        std = outputs.std(0) * 3
        pre = outputs.mean(0)
        for j in range(TEST_BATCH_SIZE):
            mdata = scio.loadmat('/home/zhengxiaohu/xx/Data/data_200*200_v2/test/' + str(j + 1) + '.mat')
            test_loss += torch.mean(torch.abs(pre[j][0].cpu() * 100 + 260 - mdata['u']))
        test_loss = test_loss / TEST_BATCH_SIZE
        break
print("Test Loss: {:.4f}".format(test_loss))

# %% md

## min MAE model

# %%

for i in range(TEST_BATCH_SIZE):
    mdata = scio.loadmat('/home/zhengxiaohu/xx/Data/data_200*200_v2/test/' + str(i + 1) + '.mat')
    X, Y = np.meshgrid(mdata['xs'][:, 0], mdata['ys'][0])

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, figsize=(20, 3), dpi=150)

    c0 = ax0.pcolormesh(data[i][0].cpu(), cmap='Greys')
    fig.colorbar(c0, ax=ax0)

    c1 = ax1.pcolormesh(X, Y, target[i][0].T.cpu() * 100 + 260, cmap='jet', vmin=298, vmax=340)
    fig.colorbar(c1, ax=ax1)

    c2 = ax2.pcolormesh(X, Y, pre[i][0].T.cpu() * 100 + 260, cmap='jet', vmin=298, vmax=340)
    fig.colorbar(c2, ax=ax2)

    c3 = ax3.pcolormesh(X, Y, (pre[i][0].cpu() * 100 + 260 - mdata['u']).abs().T, cmap='jet')  # ,vmin=0,vmax=1)
    fig.colorbar(c3, ax=ax3)

    c4 = ax4.pcolormesh(X, Y, std[i][0].cpu().T, cmap='jet')
    fig.colorbar(c4, ax=ax4)

    plt.show()

    print("Max of Error: {:.4f}, MAE: {:.4f}, Mean of 3*std: {:.4f}"
          .format((pre[i][0].cpu() * 100 + 260 - mdata['u']).abs().max(),
                  (pre[i][0].cpu() * 100 + 260 - mdata['u']).abs().mean(),
                  std[i][0].cpu().mean()))
    print(
        '=========================================================================================================================')

# %% md

## last model

# %%

for i in range(TEST_BATCH_SIZE):
    mdata = scio.loadmat('/home/zhengxiaohu/xx/Data/data_200*200_v2/test/' + str(i + 1) + '.mat')
    X, Y = np.meshgrid(mdata['xs'][:, 0], mdata['ys'][0])

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, figsize=(20, 3), dpi=150)

    c0 = ax0.pcolormesh(data[i][0].cpu(), cmap='Greys')
    fig.colorbar(c0, ax=ax0)

    c1 = ax1.pcolormesh(X, Y, target[i][0].T.cpu() * 100 + 260, cmap='jet', vmin=298, vmax=340)
    fig.colorbar(c1, ax=ax1)

    c2 = ax2.pcolormesh(X, Y, pre[i][0].T.cpu() * 100 + 260, cmap='jet', vmin=298, vmax=340)
    fig.colorbar(c2, ax=ax2)

    c3 = ax3.pcolormesh(X, Y, (pre[i][0].cpu() * 100 + 260 - mdata['u']).abs().T, cmap='jet')  # ,vmin=0,vmax=1)
    fig.colorbar(c3, ax=ax3)

    c4 = ax4.pcolormesh(X, Y, std[i][0].cpu().T, cmap='jet')
    fig.colorbar(c4, ax=ax4)

    plt.show()

    print("Max of Error: {:.4f}, MAE: {:.4f}, Mean of 3*std: {:.4f}"
          .format((pre[i][0].cpu() * 100 + 260 - mdata['u']).abs().max(),
                  (pre[i][0].cpu() * 100 + 260 - mdata['u']).abs().mean(),
                  std[i][0].cpu().mean()))
    print(
        '=========================================================================================================================')

# %% md

# 改变组件个数观察不确定性

# %%

from matplotlib import pyplot as plt
import seaborn as sns

transform_layout = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0], std=[10000])
])
transform_heat = transforms.Compose([
    transforms.ToTensor(),
    #                 transforms.Normalize(mean=[298], std=[2])
])

# net.eval()
std_list = [list() for i in range(10)]
for j in range(2, 22, 2):
    test_dataset = LayoutDataset(root='/home/zhengxiaohu/xx/Data/es/test' + str(j),
                                 train=False,
                                 transform=transform_layout,
                                 target_transform=transform_heat)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=TEST_BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=4)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_dataloader)):
            data = data.type(torch.FloatTensor)
            data = data.to(DEVICE)
            outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE, 1, 200, 200).to(DEVICE)
            for i in range(TEST_SAMPLES):
                outputs[i] = net(data, training_=True, training_mc=True)
            std = outputs.std(0) * 3
            pre = outputs.mean(0)
            std_list[j // 2 - 1].append(std.cpu().mean())

# %%

for i in range(10):
    np.save('/home/zhengxiaohu/xx/2020.8.31/data组件个数/std_list[' + str(i) + ']', np.array(std_list[i]))


# %%

def sigmoid_function(z, temp):
    fz = []
    for num in z:
        fz.append(1 / (1 + math.exp(-4 * num + 4 * temp)))
    return fz


# %%

from numpy import *

temp = mean(std_list[9])

plt.subplots(2, 5, figsize=(20, 8), dpi=100)
kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, bins=20)
plt.subplot(2, 5, 1)
plt.hist(sigmoid_function(std_list[9], temp), **kwargs)
plt.title('20-20')

plt.subplot(2, 5, 2)
plt.hist(sigmoid_function(std_list[9], temp), **kwargs)
plt.hist(sigmoid_function(std_list[8], temp), **kwargs)
plt.title('20-18')

plt.subplot(2, 5, 3)
plt.hist(sigmoid_function(std_list[9], temp), **kwargs)
plt.hist(sigmoid_function(std_list[7], temp), **kwargs)
plt.title('20-16')

plt.subplot(2, 5, 4)
plt.hist(sigmoid_function(std_list[9], temp), **kwargs)
plt.hist(sigmoid_function(std_list[6], temp), **kwargs)
plt.title('20-14')

plt.subplot(2, 5, 5)
plt.hist(sigmoid_function(std_list[9], temp), **kwargs)
plt.hist(sigmoid_function(std_list[5], temp), **kwargs)
plt.title('20-12')

plt.subplot(2, 5, 6)
plt.hist(sigmoid_function(std_list[9], temp), **kwargs)
plt.hist(sigmoid_function(std_list[4], temp), **kwargs)
plt.title('20-10')

plt.subplot(2, 5, 7)
plt.hist(sigmoid_function(std_list[9], temp), **kwargs)
plt.hist(sigmoid_function(std_list[3], temp), **kwargs)
plt.title('20-8')

plt.subplot(2, 5, 8)
plt.hist(sigmoid_function(std_list[9], temp), **kwargs)
plt.hist(sigmoid_function(std_list[2], temp), **kwargs)
plt.title('20-6')

plt.subplot(2, 5, 9)
plt.hist(sigmoid_function(std_list[9], temp), **kwargs)
plt.hist(sigmoid_function(std_list[1], temp), **kwargs)
plt.title('20-4')

plt.subplot(2, 5, 10)
plt.hist(sigmoid_function(std_list[9], temp), **kwargs)
plt.hist(sigmoid_function(std_list[0], temp), **kwargs)
plt.title('20-2')

plt.show()
# print(std_list)

# %% md

# MNIST and FashionMNIST

# %%

test_dataset = datasets.MNIST(root='/home/zhengxiaohu/Datasets',
                              download=False,
                              transform=transforms.Compose([
                                  transforms.Resize(size=(200, 200)),
                                  transforms.ToTensor()
                              ]),
                              train=False)
test_dataset, _ = torch.utils.data.random_split(test_dataset, [1000, 9000])
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=TEST_BATCH_SIZE,
                                          shuffle=False,
                                          num_workers=4)

# %%

from matplotlib import pyplot as plt
import seaborn as sns

net.eval()
std_list = [list() for i in range(3)]
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(tqdm(test_dataloader)):
        data = data.to(DEVICE)
        outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE, 1, 200, 200).to(DEVICE)
        for i in range(TEST_SAMPLES):
            outputs[i] = net(data, sample=True)
        std = outputs.var(0)
        #         pre = outputs.mean(0)
        std_list[0].append(std.cpu().mean())

# %%

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
        data = data.to(DEVICE)
        outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE, 1, 200, 200).to(DEVICE)
        for i in range(TEST_SAMPLES):
            outputs[i] = net(data, sample=True)
        std = outputs.var(0)
        #         pre = outputs.mean(0)
        std_list[1].append(std.cpu().mean())

# %%

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
        data = data.to(DEVICE)
        outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE, 1, 200, 200).to(DEVICE)
        for i in range(TEST_SAMPLES):
            outputs[i] = net(data, sample=True)
        std = outputs.var(0)
        #         pre = outputs.mean(0)
        std_list[2].append(std.cpu().mean())

# %%

plt.subplots(1, 3, figsize=(20, 8), dpi=100)

plt.subplot(1, 3, 1)
sns.rugplot(std_list[0], label='Layout', color='b', height=0.5)
plt.legend()
# plt.xlim(0,2.5)

plt.subplot(1, 3, 2)
sns.rugplot(std_list[1], label='FashionMNIST', color='g', height=0.5)
plt.legend()
# plt.xlim(0,2.5)

plt.subplot(1, 3, 3)
sns.rugplot(std_list[2], label='MNIST', color='r', height=0.5)
plt.legend()
# plt.xlim(0,2.5)

plt.show()

# %%

print(np.array(std_list[0]).var(), np.array(std_list[0]).mean())
print(np.array(std_list[1]).var(), np.array(std_list[1]).mean())
print(np.array(std_list[2]).var(), np.array(std_list[2]).mean())

# %%

plt.subplots(1, 3, figsize=(20, 8), dpi=100)

plt.subplot(1, 3, 1)
plt.boxplot(std_list[0])

plt.subplot(1, 3, 2)
plt.boxplot(std_list[1])

plt.subplot(1, 3, 3)
plt.boxplot(std_list[2])

plt.show()

# %%

sns.kdeplot(std_list[0], label='Layout', color='b')
sns.kdeplot(std_list[1], label='FashionMNIST', color='g')
sns.kdeplot(std_list[2], label='MNIST', color='r')
plt.legend()
plt.show()
