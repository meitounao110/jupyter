# !coding:utf-8
import torch
import numpy as np
from pathlib import Path
from util.loss import Jacobi_layer, LaplaceLoss, OHEMF12d
from itertools import cycle
from torch.autograd import Variable
from util.min_norm_solvers import MinNormSolver, gradient_normalizers


# from torch.distributions.categorical import Categorical
# from torch.nn import functional as F
# import matplotlib.pyplot as plt


class PseudoLabel:

    def __init__(self, model, optimizer, loss_l, device, config, writer=None, save_dir=None, save_freq=20):
        self.model = model
        self.optimizer = optimizer
        self.loss_l = loss_l
        self.loss_ul = Jacobi_layer()
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.device = device
        self.writer = writer
        self.list_path2 = config.list_path2
        self.list_path1 = config.list_path1
        self.global_step = 0  # 全局训练数据的次数
        self.epoch = 0
        self.T1, self.T2 = config.t1, config.t2
        self.af = config.af
        self.upsize = config.upsize
        self.initial_task_loss = 0

    def outputs(self, data):
        # outputs_feature = self.model(data.float())
        # outputs = F.interpolate(outputs_feature, size=(int(self.upsize), int(self.upsize)), mode='bilinear',
        #                         align_corners=True)
        outputs = data * 100 + 298
        return outputs

    def _iteration(self, data_loader, print_freq, is_train=True):
        loop_loss = []
        # accuracy = []
        # labeled_n = 0
        mode = "train" if is_train else "test"
        if type(data_loader) == dict:
            # data_loader = zip(cycle(data_loader['l']), data_loader['ul'])
            data_loader = zip(data_loader['l'], data_loader['ul'])
        for batch_idx, (data, targets) in enumerate(data_loader):  # 将可遍历的对象组成索引序列，并标出数据和下标
            self.global_step += batch_idx
            if is_train:
                if self.list_path2 != 'None' and self.list_path1 != 'None':  # 半监督
                    data_label = data[0].float()
                    targets_label = data[1].float()
                    data_unlabel = targets[0].float()
                    targets_unlabel = targets[1].float()
                    data_label, data_unlabel, targets_label, targets_unlabel = data_label.to(
                        self.device), data_unlabel.to(self.device), targets_label.to(self.device), targets_unlabel.to(
                        self.device)
                    outputs_l = self.model(data_label)
                    outputs_ul = self.model(data_unlabel)
                    # labeled_bs = self.labeled_bs
                    labeled_loss = self.loss_l(outputs_l + 298, targets_label)
                    # labeled_loss = torch.sum(self.loss_fn(outputs_l, targets_label.float())) / labeled_bs
                    with torch.no_grad():
                        x = self.loss_ul(data_unlabel * 10000, outputs_ul.detach())
                    ohem_loss1 = OHEMF12d()
                    unlabeled_loss = ohem_loss1(outputs_ul, x)
                    ####
                    loss_data = {}
                    grads = {}
                    scale = {}
                    tasks = ["1", "2"]
                    loss_t = {"1": labeled_loss, "2": unlabeled_loss}
                    # loss_t["1"] = labeled_loss
                    # loss_t["2"] = unlabeled_loss
                    # Compute gradients of each loss function wrt z
                    for t in tasks:
                        self.optimizer.zero_grad()
                        loss_data[t] = loss_t[t].data
                        loss_t[t].backward(retain_graph=True)
                        grads[t] = []
                        for param in self.model.parameters():
                            if param.grad is not None:
                                grads[t].append(Variable(param.grad.data.clone(), requires_grad=False))
                    gn = gradient_normalizers(grads, loss_data, "loss+")
                    for t in tasks:
                        for gr_i in range(len(grads[t])):
                            grads[t][gr_i] = grads[t][gr_i] / gn[t]
                    sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
                    for i, t in enumerate(tasks):
                        scale[t] = float(sol[i])
                    # self.optimizer.zero_grad()
                    for i, t in enumerate(tasks):
                        if i > 0:
                            loss = loss + scale[t] * loss_t[t]
                        else:
                            loss = scale[t] * loss_t[t]
                    ####
                    # loss = (self.model.weights[0] ** 2) * labeled_loss + (self.model.weights[1] ** 2) * unlabeled_loss
                elif self.list_path2 != 'None' and self.list_path1 == 'None':  # 有监督
                    data, targets = data.float().to(self.device), targets.float().to(
                        self.device)
                    outputs = self.model(data)
                    labeled_loss = self.loss_l(outputs + 298, targets)
                    unlabeled_loss = torch.Tensor([0])
                    loss = labeled_loss
                elif self.list_path2 == 'None' and self.list_path1 != 'None':  # 无监督
                    data, targets = data.float().to(self.device), targets.float().to(self.device)
                    outputs = self.model(data)
                    labeled_loss = torch.Tensor([0])
                    with torch.no_grad():
                        x = self.loss_ul(data * 10000, outputs.detach())
                    ohem_loss1 = OHEMF12d()
                    unlabeled_loss = ohem_loss1(outputs, x)
                    loss = unlabeled_loss
                self.optimizer.zero_grad()
                # loss.backward(retain_graph=True)
                loss.backward()
                ########
                '''
                self.model.weights.grad.data = self.model.weights.grad.data * 0.0
                norms = []
                task_loss = []
                task_loss.append(labeled_loss)
                task_loss.append(unlabeled_loss)
                task_loss = torch.stack(task_loss)
                if self.epoch == 0:
                    # set L(0)
                    if torch.cuda.is_available():
                        initial_task_loss = task_loss.data.cpu()
                    else:
                        initial_task_loss = task_loss.data
                    self.initial_task_loss = initial_task_loss.numpy()
                for i in range(len(task_loss)):
                    gygw = torch.autograd.grad(task_loss[i], self.model.model.parameters(), retain_graph=True)
                    norms.append(torch.norm(torch.mul(self.model.weights[i], gygw[0])))
                norms = torch.stack(norms)
                if torch.cuda.is_available():
                    loss_ratio = task_loss.data.cpu().numpy() / self.initial_task_loss
                else:
                    loss_ratio = task_loss.data.numpy() / self.initial_task_loss
                inverse_train_rate = loss_ratio / np.mean(loss_ratio)
                if torch.cuda.is_available():
                    mean_norm = np.mean(norms.data.cpu().numpy())
                else:
                    mean_norm = np.mean(norms.data.numpy())
                constant_term = torch.tensor(mean_norm * (inverse_train_rate ** 0.12), requires_grad=False)
                if torch.cuda.is_available():
                    constant_term = constant_term.to(self.device)
                grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
                self.model.weights.grad = torch.autograd.grad(grad_norm_loss, self.model.weights)[0]
                '''
                ########
                self.optimizer.step()
            else:
                data, targets = data.float().to(self.device), targets.float().to(
                    self.device)  # 放到GPU上
                outputs = self.model(data)
                labeled_loss = unlabeled_loss = torch.Tensor([0])
                # loss=self.loss_ul(data, outputs * 100 + 298)
                loss = self.loss_l(outputs + 298, targets)
                scale = {"1": 0, "2": 0}
            # labeled_n += labeled_bs
            loop_loss.append(loss.item())
            # acc = loss.item()
            # targets.eq(outputs.max(1)[1]).sum().item()
            # accuracy.append(acc)
            if print_freq > 0 and (batch_idx % print_freq) == 0:
                print(
                    f"[{mode}]loss[{batch_idx:<3}]\t labeled loss: {labeled_loss.item():.10f}\t unlabeled loss: {unlabeled_loss.item():.10f}\t loss: {loss.item():.10f}")
                # f"scale1:{scale['1']}\t scale2:{scale['2']}")
            if self.writer:
                self.writer.add_scalar(mode + '_global_loss', loss.item(), self.global_step)
                # self.writer.add_scalar(mode + '_global_accuracy', acc / labeled_bs, self.global_step)
        loop_loss = np.array(loop_loss).mean()
        print(f">>>[{mode}]loss\t loss: {loop_loss:.10f}")
        if self.writer:
            self.writer.add_scalar(mode + '_epoch_loss', loop_loss, self.epoch)

        return loop_loss

    # def split_list(self, data, targets, iflabel):
    #     data_label = []
    #     data_unlabel = []
    #     targets_label = []
    #     targets_unlabel = []
    #     for index in range(len(iflabel)):
    #         if iflabel[index] == 1:
    #             data_label.append(data[index:index + 1, :, :, :])
    #             targets_label.append(targets[index:index + 1, :, :, :])
    #         elif iflabel[index] == 0:
    #             data_unlabel.append(data[index:index + 1, :, :, :])
    #             targets_unlabel.append(targets[index:index + 1, :, :, :])
    #     data_label = torch.cat(data_label, dim=0)
    #     targets_label = torch.cat(targets_label, dim=0)
    #     data_unlabel = torch.cat(data_unlabel, dim=0)
    #     targets_unlabel = torch.cat(targets_unlabel, dim=0)
    #     return data_label, data_unlabel, targets_label, targets_unlabel

    def unlabeled_weight(self, loss1, loss2):
        alpha = 0.0
        # if self.epoch > self.T1:
        #     alpha = (self.epoch - self.T1) / (self.T2 - self.T1) * self.af
        #     if self.epoch > self.T2:
        #         alpha = self.af
        c1 = 0
        c2 = 0
        while loss1 < 1:
            loss1 = loss1 * 10
            c1 += 1
        while loss2 < 1:
            loss2 = loss2 * 10
            c2 += 1
        if c2 > c1:
            alpha = 10 ** (c2 - c1)
        else:
            alpha = 1
        return alpha

    def train(self, data_loader, print_freq=20):
        self.model.train()  # 启用dropout和batchnormalization
        with torch.enable_grad():
            loss = self._iteration(data_loader, print_freq)

    def test(self, data_loader, print_freq=30):
        self.model.eval()  # 不启用dropout和batchnormalization
        with torch.no_grad():
            loss = self._iteration(data_loader, print_freq, is_train=False)
        return loss

    def loop(self, epochs, train_data, test_data, scheduler=None, print_freq=-1):  # 开始循环训练，epochs 训练数据、测试数据
        minloss = 10000000
        for ep in range(epochs):
            self.epoch = ep  # ep表示当次迭代数
            print("------ Training epochs: {} ------".format(ep))
            self.train(train_data, print_freq)  # print_freq为输出的频率？
            print("------ Testing epochs: {} ------".format(ep))
            loss = self.test(test_data)
            if scheduler is not None:
                scheduler.step()
            # if ep % self.save_freq == 0:
            if loss < minloss:
                minloss = loss
                self.save(ep)

    def testonce(self, test_data, print_freq=5):
        print("------ Testing epochs: {} ------".format(1))
        loss = self.test(test_data, print_freq)

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            state = {"epoch": epoch,
                     "weight": self.model.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir()
            from datetime import datetime
            # current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            torch.save(state, model_out_path / "semi_dw1_onepoint200_500l_7500ul_netc.pth")



    def data_augment(self,data):

