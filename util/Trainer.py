#!coding:utf-8
import torch
from torch.distributions.categorical import Categorical
from torch.nn import functional as F

from pathlib import Path
from util.datasets import NO_LABEL


class PseudoLabel:

    def __init__(self, model, optimizer, loss_fn, device, config, writer=None, save_dir=None, save_freq=5):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.device = device
        self.writer = writer
        self.labeled_bs = config.labeled_batch_size  #
        self.global_step = 0  # 全局训练数据的次数
        self.epoch = 0
        self.T1, self.T2 = config.t1, config.t2
        self.af = config.af
        self.upsize = config.upsize

    def _iteration(self, data_loader, print_freq, is_train=True):
        loop_loss = []
        accuracy = []
        labeled_n = 0
        mode = "train" if is_train else "test"
        for batch_idx, (data, targets, iflabel) in enumerate(data_loader):  # 将可遍历的对象组成索引序列，并标出数据和下标
            self.global_step += batch_idx
            if is_train:
                data_label, data_unlabel, targets_label, targets_unlabel = self.split_list(data, targets, iflabel)
                data_label, data_unlabel, targets_label, targets_unlabel = data_label.to(self.device), data_unlabel.to(
                    self.device), targets_label.to(self.device), targets_unlabel.to(self.device)
                # data, targets = data.to(self.device), targets.to(self.device)  # 放到GPU上
                # 将输出插值上采样
                outputs_feature_l = self.model(data_label.float())
                outputs_l = F.interpolate(outputs_feature_l, size=(int(self.upsize), int(self.upsize)), mode='bilinear',
                                          align_corners=True)
                outputs_l = outputs_l * 50 + 298
                outputs_feature_ul = self.model(data_unlabel.float())
                outputs_ul = F.interpolate(outputs_feature_ul, size=(int(self.upsize), int(self.upsize)),
                                           mode='bilinear', align_corners=True)
                outputs_ul = outputs_ul * 50 + 298
            else:
                data, targets = data.to(self.device), targets.to(self.device)  # 放到GPU上
                outputs_feature = self.model(data.float())
                outputs = F.interpolate(outputs_feature, size=(int(self.upsize), int(self.upsize)), mode='bilinear',
                                        align_corners=True)
                outputs = outputs * 50 + 298

            if is_train:  # 训练时
                labeled_bs = self.labeled_bs
                labeled_loss = self.loss_fn(outputs_l, targets_label.float())
                # labeled_loss = torch.sum(self.loss_fn(outputs_l, targets_label.float())) / labeled_bs
                with torch.no_grad():
                    outputs_feature_ulp = self.model(data_unlabel.float())
                    outputs_ulp = F.interpolate(outputs_feature_ulp, size=(int(self.upsize), int(self.upsize)),
                                                mode='bilinear',
                                                align_corners=True)
                    outputs_ulp = outputs_ulp * 50 + 298
                    pseudo_labeled = outputs_ulp
                # unlabeled_loss = self.loss_fn(outputs_ul, pseudo_labeled)
                unlabeled_loss = self.loss_fn(outputs_ul, pseudo_labeled.float())
                loss = labeled_loss + self.unlabeled_weight() * unlabeled_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                labeled_bs = data.size(0)
                labeled_loss = unlabeled_loss = torch.Tensor([0])
                loss = self.loss_fn(outputs, targets.float())
            labeled_n += labeled_bs

            loop_loss.append(loss.item() / len(data_loader))
            acc = loss.item()
            # targets.eq(outputs.max(1)[1]).sum().item()
            accuracy.append(acc)
            if print_freq > 0 and (batch_idx % print_freq) == 0:
                print(
                    f"[{mode}]loss[{batch_idx:<3}]\t labeled loss: {labeled_loss.item():.3f}\t unlabeled loss: {unlabeled_loss.item():.3f}\t loss: {loss.item():.3f}\t Acc: {acc / labeled_bs:.3%}")
            if self.writer:
                self.writer.add_scalar(mode + '_global_loss', loss.item(), self.global_step)
                # self.writer.add_scalar(mode + '_global_accuracy', acc / labeled_bs, self.global_step)
        print(f">>>[{mode}]loss\t loss: {sum(loop_loss):.3f}\t Acc: {sum(accuracy) / labeled_n:.3%}")
        if self.writer:
            self.writer.add_scalar(mode + '_epoch_loss', sum(loop_loss), self.epoch)
            self.writer.add_scalar(mode + '_epoch_accuracy', sum(accuracy) / labeled_n, self.epoch)

        return loop_loss, accuracy

    def split_list(self, data, targets, iflabel):
        data_label = []
        data_unlabel = []
        targets_label = []
        targets_unlabel = []
        for index in range(len(iflabel)):
            if iflabel[index] == 1:
                data_label.append(data[index:index + 1, :, :, :])
                targets_label.append(targets[index:index + 1, :, :, :])
            elif iflabel[index] == 0:
                data_unlabel.append(data[index:index + 1, :, :, :])
                targets_unlabel.append(targets[index:index + 1, :, :, :])
        data_label = torch.cat(data_label, dim=0)
        targets_label = torch.cat(targets_label, dim=0)
        data_unlabel = torch.cat(data_unlabel, dim=0)
        targets_unlabel = torch.cat(targets_unlabel, dim=0)
        return data_label, data_unlabel, targets_label, targets_unlabel

    def unlabeled_weight(self):
        alpha = 0.0
        if self.epoch > self.T1:
            alpha = (self.epoch - self.T1) / (self.T2 - self.T1) * self.af
            if self.epoch > self.T2:
                alpha = self.af
        return alpha

    def train(self, data_loader, print_freq=20):
        self.model.train()  # 启用dropout和batchnormalization
        with torch.enable_grad():
            loss, correct = self._iteration(data_loader, print_freq)

    def test(self, data_loader, print_freq=10):
        self.model.eval()  # 不启用dropout和batchnormalization
        with torch.no_grad():
            loss, correct = self._iteration(data_loader, print_freq, is_train=False)

    def loop(self, epochs, train_data, test_data, scheduler=None, print_freq=-1):  # 开始循环训练，epochs 训练数据、测试数据
        for ep in range(epochs):
            self.epoch = ep  # ep表示当次迭代数
            print("------ Training epochs: {} ------".format(ep))
            self.train(train_data, print_freq)  # print_freq为输出的频率？
            print("------ Testing epochs: {} ------".format(ep))
            self.test(test_data, print_freq)
            if scheduler is not None:
                scheduler.step()
            if ep % self.save_freq == 0:
                self.save(ep)

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            state = {"epoch": epoch,
                     "weight": self.model.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir()
            torch.save(state, model_out_path / "model_epoch_{}.pth".format(epoch))
