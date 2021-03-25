import torch
from torch.nn import MSELoss
import torch.nn.functional as F
from torch.nn import _reduction as _Reduction
from torch.nn.modules.module import Module


class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class Outsideloss(_Loss):
    def __init__(self, base_loss=MSELoss, length=1, u_D=298, bcs=None, nx=200):
        super().__init__()
        self.base_loss = base_loss
        self.u_D = u_D  # 最低温
        self.slice_bcs = []  #
        self.bcs = bcs  # 小孔
        self.nx = nx  # 划分网格
        self.length = length  # 边长

    def forward(self, inputs):
        if self.bcs is None:  # 全边散热，all bcs are Dirichlet
            d1 = inputs[:, :, :1, :]  # 第一行
            d2 = inputs[:, :, -1:, :]  # 倒数第一行
            d3 = inputs[:, :, 1:-1, :1]  # 第二行到倒数第二行，最左边
            d4 = inputs[:, :, 1:-1, -1:]  # 第二行到倒数第二行，最右边
            point = torch.cat([d1.flatten(), d2.flatten(), d3.flatten(), d4.flatten()], dim=0)
            return self.base_loss(point, torch.ones_like(point) * 0)
        loss = 0
        for bc in self.bcs:
            if bc[0][1] == 0 and bc[1][1] == 0:
                idx_start = round(bc[0][0] * self.nx / self.length)
                idx_end = round(bc[1][0] * self.nx / self.length)
                point = inputs[..., idx_start:idx_end, :1]
                loss += self.base_loss(point, torch.ones_like(point) * 0)
                # print('2*************************')
            elif bc[0][1] == self.length and bc[1][1] == self.length:
                idx_start = round(bc[0][0] * self.nx / self.length)
                idx_end = round(bc[1][0] * self.nx / self.length)
                point = inputs[..., idx_start:idx_end, -1:]
                loss += self.base_loss(point, torch.ones_like(point) * 0)
                # print('3*************************')
            elif bc[0][0] == 0 and bc[1][0] == 0:
                idx_start = round(bc[0][1] * self.nx / self.length)
                idx_end = round(bc[1][1] * self.nx / self.length)
                point = inputs[..., :1, idx_start:idx_end]
                loss += self.base_loss(point, torch.ones_like(point) * 0)
            elif bc[0][0] == self.length and bc[1][0] == self.length:
                idx_start = round(bc[0][1] * self.nx / self.length)
                idx_end = round(bc[1][1] * self.nx / self.length)
                point = inputs[..., -1:, idx_start:idx_end]
                loss += self.base_loss(point, torch.ones_like(point) * 0)
            else:
                raise ValueError("bc error!")
        return loss


# class LaplaceLoss(_Loss):
#     def __init__(
#             self, base_loss=MSELoss(reduction='mean'), nx=200,
#             length=0.1, weight=[[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], bcs=[[[0.0495, 0], [0.0505, 0]]],
#     ):
#         super().__init__()
#         self.base_loss = base_loss
#         self.weight = torch.Tensor(weight)
#         self.bcs = bcs
#         self.length = length
#         self.nx = nx
#         self.scale_factor = 1  # self.nx/200
#         TEMPER_COEFFICIENT = 100  # 50.0
#         STRIDE = self.length / (self.nx-1)
#         self.cof = -1 * STRIDE ** 2 / TEMPER_COEFFICIENT
#
#     def laplace(self, x):
#         return F.conv2d(x, self.weight.to(device=x.device), bias=None, stride=1, padding=0)
#
#     def forward(self, layout, heat):
#         #layout = interpolate(layout, scale_factor=self.scale_factor)
#
#         heat = F.pad(heat, [1, 1, 1, 1], mode='reflect')  # constant, reflect, replicate
#         layout_pred = self.laplace(heat)
#         if self.bcs is None or len(self.bcs) == 0 or len(self.bcs[0]) == 0:  # all are Dirichlet bcs
#             return self.base_loss(layout_pred[..., 1:-1, 1:-1], self.cof * layout[..., 1:-1, 1:-1])
#         else:
#             for bc in self.bcs:
#                 if bc[0][1] == 0 and bc[1][1] == 0:
#                     idx_start = round(bc[0][0] * self.nx / self.length)
#                     idx_end = round(bc[1][0] * self.nx / self.length) + 1
#                     layout_pred[..., idx_start:idx_end, :1] = self.cof * layout[..., idx_start:idx_end, :1]
#                 elif bc[0][1] == self.length and bc[1][1] == self.length:
#                     idx_start = round(bc[0][0] * self.nx / self.length)
#                     idx_end = round(bc[1][0] * self.nx / self.length)
#                     layout_pred[..., idx_start:idx_end, -1:] = self.cof * layout[..., idx_start:idx_end, -1:]
#                 elif bc[0][0] == 0 and bc[1][0] == 0:
#                     idx_start = round(bc[0][1] * self.nx / self.length)
#                     idx_end = round(bc[1][1] * self.nx / self.length)
#                     layout_pred[..., :1, idx_start:idx_end] = self.cof * layout[..., :1, idx_start:idx_end]
#                 elif bc[0][0] == self.length and bc[1][0] == self.length:
#                     idx_start = round(bc[0][1] * self.nx / self.length)
#                     idx_end = round(bc[1][1] * self.nx / self.length)
#                     layout_pred[..., -1:, idx_start:idx_end] = self.cof * layout[..., -1:, idx_start:idx_end]
#                 else:
#                     raise ValueError("bc error!")
#         return self.base_loss(layout_pred, self.cof * layout)#,layout_pred/(self.cof)


class LaplaceLoss(_Loss):
    def __init__(
            self,
            # base_loss=MSELoss(reduction='mean'),
            nx=50,
            length=0.1,
            weight=[[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]],
            bcs=[[[0.0, 0], [0.1, 0]]],  # [[[0.0495, 0], [0.0505, 0]]],
    ):
        super().__init__()
        # self.base_loss = base_loss
        self.weight = torch.Tensor(weight)  # (
        # torch.tensor(weight).float().unsqueeze(0).unsqueeze(0)
        # )  # shape (1, 1, m, n)
        self.bcs = bcs
        self.length = length
        self.nx = nx
        # self.scale_factor = self.nx / 200
        TEMPER_COEFFICIENT = 100
        STRIDE = self.length / (self.nx - 1)
        # self.cof = -(STRIDE ** 2) * 10000 / TEMPER_COEFFICIENT
        self.cof = (STRIDE ** 2) * 10000

    def laplace(self, x):
        return F.conv2d(x, self.weight.to(device=x.device), bias=None, stride=1, padding=0)

    def forward(self, layout, heat):
        # layout = F.interpolate(layout, scale_factor=self.scale_factor)
        # print(layout_pred.size())
        G = torch.ones_like(heat)
        if (
                self.bcs is None or len(self.bcs) == 0 or len(self.bcs[0]) == 0
        ):  # all are Dirichlet bcs
            # print('11*************************')
            G[..., 0:, :1] = torch.zeros_like(G[..., 0:, :1])
            heat = F.pad(heat * G, [1, 1, 1, 1], mode='reflect')
            layout_pred = -(self.laplace(heat) / (self.cof))
            layout_pred[..., 0:, :1] = layout[..., 0:, :1]
        else:
            for bc in self.bcs:
                if bc[0][1] == 0 and bc[1][1] == 0:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    heat[..., idx_start:idx_end, :1] = torch.zeros_like(
                        heat[..., idx_start:idx_end, :1])
                    heat = F.pad(heat, [1, 1, 1, 1], mode='reflect')
                    layout_pred = -(self.laplace(heat) / (self.cof))
                    layout_pred[..., idx_start:idx_end, :1] = layout[..., idx_start:idx_end, :1]
                    # print('22*************************')
                elif bc[0][1] == self.length and bc[1][1] == self.length:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    heat[..., idx_start:idx_end, -1:] = torch.zeros_like(
                        heat[..., idx_start:idx_end, -1:])
                    heat = F.pad(heat, [1, 1, 1, 1], mode='reflect')
                    layout_pred = -(self.laplace(heat) / (self.cof))
                    layout_pred[..., idx_start:idx_end, -1:] = layout[..., idx_start:idx_end, -1:]
                    # print('33*************************')
                elif bc[0][0] == 0 and bc[1][0] == 0:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    heat[..., :1, idx_start:idx_end] = torch.zeros_like(
                        heat[..., :1, idx_start:idx_end])
                    heat = F.pad(heat, [1, 1, 1, 1], mode='reflect')
                    layout_pred = -(self.laplace(heat) / (self.cof))
                    layout_pred[..., :1, idx_start:idx_end] = layout[..., :1, idx_start:idx_end]
                    # print(idx_start)
                    # print(idx_end)
                    # print(layout_pred[..., :1, idx_start:idx_end].size())
                    # print('44*************************')
                elif bc[0][0] == self.length and bc[1][0] == self.length:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    heat[..., -1:, idx_start:idx_end] = torch.zeros_like(
                        heat[..., -1:, idx_start:idx_end])
                    heat = F.pad(heat, [1, 1, 1, 1], mode='reflect')
                    layout_pred = -(self.laplace(heat) / (self.cof))
                    layout_pred[..., -1:, idx_start:idx_end] = layout[..., -1:, idx_start:idx_end]
                    # print('55*************************')
                else:
                    raise ValueError("bc error!")

        return layout_pred  # self.base_loss(layout_pred, layout)


class Jacobi_layer(torch.nn.Module):
    def __init__(
            self, nx=200, length=0.1, bcs=[[[0.0495, 0], [0.0505, 0]]]
    ):
        super(Jacobi_layer, self).__init__()
        self.length = length
        self.bcs = bcs
        # 雅克比迭代的权重 1/4(u_(i, j-1), u_(i, j+1), u_(i-1, j), u_(i+1, j))
        self.weight = torch.Tensor([[[[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]]]])
        # 填充
        self.nx = nx
        self.scale_factor = 1  # self.nx/200
        TEMPER_COEFFICIENT = 1  # 50
        STRIDE = self.length / (self.nx - 1)
        # ((l/(nx))^2)/(4*cof)*m*input(x, y)
        self.cof = 0.25 * STRIDE ** 2 / TEMPER_COEFFICIENT

    def jacobi(self, x):
        return F.conv2d(x, self.weight.to(device=x.device), bias=None, stride=1, padding=0)

    def forward(self, layout, heat, n_iter=1):
        # 右端项
        f = self.cof * layout
        # G: 是否为非边界点
        G = torch.ones_like(heat)

        if self.bcs is None or len(self.bcs) == 0 or len(self.bcs[0]) == 0:  # all are Dirichlet bcs
            pass
        else:
            for bc in self.bcs:
                if bc[0][1] == 0 and bc[1][1] == 0:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    G[..., idx_start:idx_end, :1] = torch.zeros_like(G[..., idx_start:idx_end, :1])
                elif bc[0][1] == self.length and bc[1][1] == self.length:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    G[..., idx_start:idx_end, -1:] = torch.zeros_like(G[..., idx_start:idx_end, -1:])
                elif bc[0][0] == 0 and bc[1][0] == 0:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    G[..., :1, idx_start:idx_end] = torch.zeros_like(G[..., :1, idx_start:idx_end])
                elif bc[0][0] == self.length and bc[1][0] == self.length:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    G[..., -1:, idx_start:idx_end] = torch.zeros_like(G[..., -1:, idx_start:idx_end])
                else:
                    raise ValueError("bc error!")
        for i in range(n_iter):
            if i == 0:
                x = F.pad(heat * G, [1, 1, 1, 1], mode='reflect')
            else:
                x = F.pad(x, [1, 1, 1, 1], mode='reflect')
            x = G * (self.jacobi(x) + f)
        return x


class OHEMF12d(torch.nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(self, weight=None):
        super(OHEMF12d, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets):
        diff = F.l1_loss(inputs, targets, reduction='none').detach()
        min, max = torch.min(diff.view(diff.shape[0], -1), dim=1)[0], torch.max(diff.view(diff.shape[0], -1), dim=1)[0]
        min, max = min.reshape(diff.shape[0], 1, 1, 1).expand(diff.shape), max.reshape(diff.shape[0], 1, 1, 1).expand(
            diff.shape)
        diff = 10 * (diff - min) / max
        return torch.mean(torch.abs(diff * (inputs - targets)))
