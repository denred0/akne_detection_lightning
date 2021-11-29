import torch
from torch import nn

import numpy as np

import pytorch_lightning as pl

from resnet50 import resnet50
from utils.genLD import genLD
from utils.utils import Logger, AverageMeter, time_to_str, weights_init
from utils.report import report_precision_se_sp_yi, report_mae_mse


class Model(pl.LightningModule):
    def __init__(self, sigma, lam, learning_rate=0.0001):
        super().__init__()

        # log hyperparameters
        # self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.sigma = sigma
        self.lam = lam

        self.model = resnet50().cuda()
        self.loss_func = nn.CrossEntropyLoss().cuda()
        self.kl_loss_1 = nn.KLDivLoss().cuda()
        self.kl_loss_2 = nn.KLDivLoss().cuda()
        self.kl_loss_3 = nn.KLDivLoss().cuda()

        self.losses_cls = AverageMeter()
        self.losses_cou = AverageMeter()
        self.losses_cou2cls = AverageMeter()
        self.losses = AverageMeter()

    # will be used during inference
    def forward(self, x):
        return self.model(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        b_x, b_y, b_l = batch

        b_x = b_x.cuda()
        b_l = b_l.cpu().numpy()

        # generating ld
        b_l = b_l - 1
        ld = genLD(b_l, self.sigma, 'klloss', 65)
        ld_4 = np.vstack((np.sum(ld[:, :5], 1), np.sum(ld[:, 5:20], 1), np.sum(ld[:, 20:50], 1),
                          np.sum(ld[:, 50:], 1))).transpose()
        ld = torch.from_numpy(ld).cuda().float()
        ld_4 = torch.from_numpy(ld_4).cuda().float()

        cls, cou, cou2cls = self.model(b_x, None)
        loss_cls = self.kl_loss_1(torch.log(cls), ld_4) * 4.0
        loss_cou = self.kl_loss_2(torch.log(cou), ld) * 65.0
        loss_cls_cou = self.kl_loss_3(torch.log(cou2cls), ld_4) * 4.0
        loss = (loss_cls + loss_cls_cou) * 0.5 * self.lam + loss_cou * (1.0 - self.lam)

        self.losses_cls.update(loss_cls.item(), b_x.size(0))
        self.losses_cou.update(loss_cou.item(), b_x.size(0))
        self.losses_cou2cls.update(loss_cls_cou.item(), b_x.size(0))
        self.losses.update(loss.item(), b_x.size(0))

        self.log('losses_cls', self.losses_cls.avg, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('losses_cou', self.losses_cou.avg, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('losses_cou2cls', self.losses_cou2cls.avg, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('losses', self.losses.avg, on_step=True, on_epoch=True, logger=True, prog_bar=True)

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        test_x, test_y, test_l = batch
        test_x = test_x.cuda()
        test_y = test_y.cuda()

        y_true = test_y.data.cpu().numpy()
        l_true = test_l.data.cpu().numpy()

        cls, cou, cou2cls = self.model(test_x, None)

        loss = self.loss_func(cou2cls, test_y)
        test_loss = loss.data

        _, preds_m = torch.max(cls + cou2cls, 1)
        _, preds = torch.max(cls, 1)
        y_pred = preds.data.cpu().numpy()
        y_pred_m = preds_m.data.cpu().numpy()

        _, preds_l = torch.max(cou, 1)
        preds_l = (preds_l + 1).data.cpu().numpy()
        l_pred = preds_l

        batch_corrects = torch.sum((preds == test_y)).data.cpu().numpy()
        test_corrects = batch_corrects

        test_loss = test_loss.float() / len(preds)
        test_acc = test_corrects / len(preds)

        self.log('test_loss', test_loss.data, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_acc', test_acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        _, AVE_ACC, pre_se_sp_yi_report = report_precision_se_sp_yi(y_pred, y_true)
        _, AVE_ACC_m, pre_se_sp_yi_report_m = report_precision_se_sp_yi(y_pred_m, y_true)
        _, MAE, MSE, mae_mse_report = report_mae_mse(l_true, l_pred, y_true)

        self.log('AVE_ACC', AVE_ACC, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('AVE_ACC_m', AVE_ACC_m, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('MAE', MAE, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('MSE', MSE, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        return test_loss

    # # logic for a single testing step
    def test_step(self, batch, batch_idx):
        test_x, test_y, test_l = batch
        test_x = test_x.cuda()
        test_y = test_y.cuda()

        y_true = test_y.data.cpu().numpy()
        l_true = test_l.data.cpu().numpy()

        cls, cou, cou2cls = self.model(test_x, None)

        loss = self.loss_func(cou2cls, test_y)
        test_loss = loss.data

        _, preds_m = torch.max(cls + cou2cls, 1)
        _, preds = torch.max(cls, 1)
        y_pred = preds.data.cpu().numpy()
        y_pred_m = preds_m.data.cpu().numpy()

        _, preds_l = torch.max(cou, 1)
        preds_l = (preds_l + 1).data.cpu().numpy()
        l_pred = preds_l

        batch_corrects = torch.sum((preds == test_y)).data.cpu().numpy()
        test_corrects = batch_corrects

        test_loss = test_loss.float() / len(preds)
        test_acc = test_corrects / len(preds)

        self.log('test_loss', test_loss.data, on_step=True, on_epoch=True, logger=True)
        self.log('test_acc', test_acc, on_step=True, on_epoch=True, logger=True)

        _, _, pre_se_sp_yi_report = report_precision_se_sp_yi(y_pred, y_true)
        _, _, pre_se_sp_yi_report_m = report_precision_se_sp_yi(y_pred_m, y_true)
        _, MAE, MSE, mae_mse_report = report_mae_mse(l_true, l_pred, y_true)

        # self.logger.write(str(pre_se_sp_yi_report) + '\n')
        # self.logger.write(str(pre_se_sp_yi_report_m) + '\n')
        # self.logger.write(str(mae_mse_report) + '\n')
        return test_loss

    def training_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)

        scheduler = {'scheduler':
                         torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999, verbose=False),
                     'interval': 'step'}  # called after each training step

        return [optimizer], [scheduler]

    # def configure_optimizers(self):
    #     optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=2e-5)
    #     lr_scheduler = PolynomialLRDecay(optimizer, max_decay_steps=50,
    #                                      end_learning_rate=1e-6,
    #                                      power=0.9, verbose=True)
    #
    #     # optimizer = MADGRAD(self.parameters(), lr=self.lr, weight_decay=2.5e-5)
    #     # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     #     optimizer, milestones=[30, 60, 90], gamma=0.1, verbose=True)
    #
    #     # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     #     optimizer=optimizer,
    #     #     factor=0.5,
    #     #     threshold=0.01,
    #     #     threshold_mode='rel',
    #     #     cooldown=3,
    #     #     mode='max',
    #     #     min_lr=1e-6,
    #     #     verbose=True,
    #     #     )
    #
    #     lr_dict = {
    #         'scheduler': lr_scheduler,
    #         'reduce_on_plateau': False,
    #         'monitor': 'val_f1_epoch',
    #         'interval': 'epoch',
    #         'frequency': 1,
    #     }
    #
    #     return [optimizer], [lr_dict]
