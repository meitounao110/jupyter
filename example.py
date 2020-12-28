import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io as sio
import torch.nn.functional as F

# data = sio.loadmat(
#     "/mnt/share1/layout_data/v0.3/data/all_walls/train/train/Example0.mat"
# )
# u_true = data["u"]  # 真实温度
# F = data["F"]  # 布局
# a=torch.Tensor([[1,2,3,4],[1,2,3,4],[1,2,3,4],[2,2,3,4]])
# print(a[-1])
# print(torch.ones_like(a) * 0)
# c=torch.randn(4,4)
# print(c)
# b=torch.Tensor([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
# weight=torch.Tensor([[0,1,0],[1,-4,1],[0,1,0]])
# weight=weight.unsqueeze(0).unsqueeze(0)
# print(weight[...,1:])
# conv=F.conv2d(c.unsqueeze(0).unsqueeze(0),weight,stride=1)
# a=u_true.tolist()
# print(min(a))
# a=torch.rand(4,3,4,4)
# print(a)
# print(a[:2],a[:2].size())
# print(a[:2,:1,:,:])
# print(a[:2,1:,:,:])

a=np.array([[1,2],[3,4]])
b=np.array([[3,3],[4,4]])

loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)

input = torch.autograd.Variable(torch.from_numpy(a))
target = torch.autograd.Variable(torch.from_numpy(b))

loss = loss_fn(input.float(), target.float())

print(loss)