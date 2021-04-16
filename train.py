import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
import torch.nn as nn
from sklearn.metrics import confusion_matrix
# from eval.confusion_matrix import base_confusion
from loss import loss
from network.ce_net import CEnet
from network.scaa import SCAA, SCAA3D
from opt import get_opts
from utils import load_ckpt
from dataset.dataset import train_dataloader, ChaoDataset, WholeDataset

import pytorch_lightning as pl
from torch import optim
from torch.utils.data import random_split
from torchvision import transforms
import os


class TrainSystem(pl.LightningModule):
    def __init__(self, param):
        super(TrainSystem, self).__init__()
        self.hparams = param
        self.n_train = None
        self.n_val = None
        self.n_classes = 1
        self.n_channels = 1
        ###############################################################################################
        # if network is unet then must use F.binary_cross_entropy_with_logits
        # worry???????
        ###############################################################################################
        # self.criterion = F.binary_cross_entropy_with_logits
        # self.model = UNet(n_channels=1,
        #                   n_classes=1,
        #                   bilinear=False,
        #                   )
        # self.model = Unet(1, 1)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # self.model = CEnet()
        self.model = SCAA(num_features=8)
        self.model3d = SCAA3D(num_features=8)
        self.criterion = nn.CrossEntropyLoss() if self.model.n_classes > 1 else nn.BCELoss()
        self.all_train = None
        self.epoch_loss = 0
        self.val = {}
        self.iou_sum = 0
        self.dice_sum = 0
        self.f3d = None
        self.sdu = self.scheduler()
        # to unnormalize image for visualization
        self.unpreprocess = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((-1.5,), (1.0,))
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=1).cuda()
        self.loss_seg_CE = loss.CELoss4MOTS(num_classes=1, ignore_index=255).cuda()
        # self.loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=self.hparams.n_classes).cuda()
        # self.loss_seg_CE = loss.CELoss4MOTS(num_classes=self.hparams.n_classes, ignore_index=255).cuda()
        self._generator = self.__dataloader()
        self._data = self.refresh_dataloader()
        # self._data = None

        # model

        # device gpu number
        if self.hparams.num_gpus == 1:
            print('number of parameters : %.2f M' %
                  (sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6))

        # load model checkpoint path is provided
        if self.hparams.ckpt_path != '':
            print('Load model from', self.hparams.ckpt_path)
            load_ckpt(self.model, self.hparams.ckpt_path, self.hparams.prefixes_to_ignore)

    def forward(self, x, pre_train_feature, task_id):
        return self.model.forward(x, pre_train_feature, task_id)

    def on_train_epoch_start(self):
        self.epoch_loss = 0

        super(TrainSystem, self).on_train_epoch_start()
        self._data = self.refresh_dataloader()
        model = self.trainer.get_model()
        self.trainer.reset_train_dataloader(model)
        # a = 1


    # def train_all(self, x):
    #     x, y = x.values()

    def training_step(self, batch, batch_nb):
        a = self.all_train
        f3d = None
        for i in a:
            x = i['image'].transpose(0, 1).unsqueeze(0).cuda()
            y = i['mask'].transpose(0, 1).unsqueeze(0).cuda()
            # f3d = self.model3d(self.all_dataloader()['train_all'])
            f3d = self.model3d(x)

        inputs, labels, _ = batch.values()
        pred = self.forward(inputs, f3d, task_id=batch['files'][batch_nb]['task_id'])

        term_seg_Dice = self.loss_seg_DICE.forward(pred, labels)
        term_seg_BCE = self.loss_seg_CE.forward(pred, labels)
        term_all = term_seg_Dice + term_seg_BCE

        reduce_Dice = torch.mean(term_seg_Dice)
        reduce_BCE = torch.mean(term_seg_BCE)
        reduce_all = torch.mean(term_all)
        # if self.hparams.FP16:
        #     with amp.scale_loss(term_all, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     term_all.backward()

        # term_all.backward()
        loss = term_all
        # loss = self.criterion(y_hat, y)
        # loss.backward()
        # loss = calc_loss(y_hat, y)

        self.log('Loss/train', loss.item(), on_step=True, on_epoch=True)
        # self.logger.experiment.add_image('y_hat', y_hat, 0)

        tensorboard_logs = {'train_loss': loss}
        self.epoch_loss += loss.item()
        return {'loss': loss, 'log': tensorboard_logs}

    def on_train_epoch_end(self, outputs) -> None:

        # self.reset_train_dataloader()
        # for tag, value in self.model.named_parameters():
        #     tag = tag.replace('.', '/')
        #     new
        # self.logger.experiment.add_histogram('weights' + tag, value.data.cpu().numpy(), self.global_step)
        # self.logger.experiment.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), self.global_step)
        super(TrainSystem, self).on_train_epoch_end(outputs)
        # self._data = self.refresh_dataloader()

    def on_validation_epoch_start(self) -> None:
        self.val = {
            'DICE': 0,
            'ACC': 0,
            'PPV': 0,
            'TPR': 0,
            'TNR': 0,
            'F1': 0,
            'LOSS': 0,
        }
        super(TrainSystem, self).on_validation_epoch_start()
        a = self.all_train
        for i in a:
            x = i['image'].transpose(0, 1).unsqueeze(0).cuda()
            y = i['mask'].transpose(0, 1).unsqueeze(0).cuda()
            # f3d = self.model3d(self.all_dataloader()['train_all'])
            self.f3d = self.model3d(x)

    def validation_step(self, batch, batch_idx):

        img, label, _ = batch.values()
        output = self.forward(img, self.f3d, task_id=batch['files'][batch_idx]['task_id'])
        # loss = self.criterion(output, label)

        term_seg_Dice = self.loss_seg_DICE.forward(output, label)
        term_seg_BCE = self.loss_seg_CE.forward(output, label)
        term_all = term_seg_Dice + term_seg_BCE

        reduce_Dice = torch.mean(term_seg_Dice)
        reduce_BCE = torch.mean(term_seg_BCE)
        reduce_all = torch.mean(term_all)
        # if self.hparams.FP16:
        #     with amp.scale_loss(term_all, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     term_all.backward()

        # term_all.backward()
        loss = term_all

        # optimizer.step()
        log = {'val_loss': loss}

        perd = output > 0.5

        ################################################################################################################
        eps = 0.0001
        # inter = torch.dot(label.view(-1), output.view(-1))  # 求交集，数据全部拉成一维 然后就点积
        # union = torch.sum(label) + torch.sum(output) + eps  # 求并集，数据全部求和后相加，eps防止分母为0
        # iou = (inter.float() + eps) / (union.float() - inter.float())  # iou计算公式，交集比上并集
        #
        # dice = (2 * inter.float() + eps) / union.float()
        # confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true=label.view(-1).cpu(),
                                          y_pred=perd[:, 1].float().view(-1).cpu()).ravel()

        self.val['LOSS'] += loss
        # iou
        # self.val['IOU'] += iou
        # dice
        dice = (2 * tp) / (fp + 2 * tp + fn + eps)
        # diceLoss = 1 - dice
        self.val['DICE'] += dice
        # ACC = (TP + TN) / (TP + TN + FP + FN)
        acc = (tp + tn) / (tp + tn + fp + fn + eps)
        self.val['ACC'] += acc
        # PPV(Precision) = TP / (TP + FP)
        ppv = tp / (tp + fp + eps)
        self.val['PPV'] += ppv
        # TPR(Sensitivity=Recall) = TP / (TP + FN)
        tpr = tp / (tp + fn + eps)
        self.val['TPR'] += tpr
        # TNR(Specificity) = TN / (TN + FP)
        tnr = tn / (tn + fp + eps)
        self.val['TNR'] += tnr
        # F1 = 2PR / (P + R)
        f1 = 2 * ppv * tpr / (ppv + tpr + eps)
        self.val['F1'] += f1
        ################################################################################################################
        self.logger.experiment.add_images('images', img, self.global_step)
        if self.model.n_classes == 1:
            self.logger.experiment.add_images('masks/true', label, self.global_step)
            self.logger.experiment.add_images('masks/pred', output > 0.5, self.global_step)

        return {'log': log}

    def on_validation_epoch_end(self):
        # 应四舍五入
        percent = (self.n_val + self.n_train) / self.hparams.batch * self.hparams.val_percent // self.hparams.num_gpus
        val_score = self.val['ACC'] / percent
        # self.log('iou', val_score, on_step=False, on_epoch=True)
        self.sdu.step(val_score)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], on_step=False, on_epoch=True)
        if self.model.n_classes > 1:
            # lg.info('validation cross entropy: {}'.format(val_score))
            self.log('Dice/test/epoch', self.val['DICE'] * percent / self.n_val, on_step=False, on_epoch=True)
            self.log('IOU', val_score, on_step=False, on_epoch=True)
            self.log('Dice/test', self.val['DICE'] / percent, on_step=False, on_epoch=True)
            self.log('ACC/test', self.val['ACC'] / percent, on_step=False, on_epoch=True)
            self.log('PPV/test', self.val['PPV'] / percent, on_step=False, on_epoch=True)
            self.log('TPR/test', self.val['TPR'] / percent, on_step=False, on_epoch=True)
            self.log('TNR/test', self.val['TNR'] / percent, on_step=False, on_epoch=True)
            self.log('F1/test', self.val['F1'] / percent, on_step=False, on_epoch=True)
        else:
            # lg.info('validation Dice Coeff: {}'.format(val_score))
            self.log('Dice/test/epoch', self.val['DICE'] * percent / self.n_val, on_step=False, on_epoch=True)
            self.log('IOU', val_score, on_step=False, on_epoch=True)
            self.log('Dice/test', self.val['DICE'] / percent, on_step=False, on_epoch=True)
            self.log('ACC/test', self.val['ACC'] / percent, on_step=False, on_epoch=True)
            self.log('PPV/test', self.val['PPV'] / percent, on_step=False, on_epoch=True)
            self.log('TPR/test', self.val['TPR'] / percent, on_step=False, on_epoch=True)
            self.log('TNR/test', self.val['TNR'] / percent, on_step=False, on_epoch=True)
            self.log('F1/test', self.val['F1'] / percent, on_step=False, on_epoch=True)

    def __dataloader(self, imgs_dir=None, masks_dir=None):
        img_list = ChaoDataset.get_list(imgs_dir=self.hparams.imgs_dir, masks_dir=self.hparams.masks_dir)
        len_dir = len(img_list)
        # for i in range(len_dir):
        i = 0
        while i < len_dir:
            transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((-1.5,), (1.0,))
                transforms.Normalize((0.5,), (0.5,))
                # transforms.Normalize(0.5, 0.5)
            ])
            target_transform = transforms.Compose([transforms.ToTensor()])
            if imgs_dir is not None and masks_dir is not None:
                # dataset = BaseDataset(imgs_dir=imgs_dir, masks_dir=masks_dir, transform=transform,
                #                       target_transform=target_transform)
                dataset = ChaoDataset(imgs_dir=self.hparams.imgs_dir, masks_dir=self.hparams.masks_dir,
                                      transform=transform,
                                      target_transform=target_transform, index=img_list[i])

                # dataset = WholeDataset(imgs_dir=self.hparams.imgs_dir, masks_dir=self.hparams.masks_dir,
                #                        transform=transform,
                #                        target_transform=target_transform)
            else:
                # transform should be given by class hparams

                # dataset = BaseDataset(imgs_dir=self.hparams.imgs_dir, masks_dir=self.hparams.masks_dir, transform=transform,
                #                       target_transform=target_transform)

                # dataset = WholeDataset(imgs_dir=self.hparams.imgs_dir, masks_dir=self.hparams.masks_dir,
                #                        transform=transform,
                #                        target_transform=target_transform)
                dataset = ChaoDataset(imgs_dir=self.hparams.imgs_dir, masks_dir=self.hparams.masks_dir,
                                      transform=transform,
                                      target_transform=target_transform, index=img_list[i])

            n_val = int(len(dataset) * self.hparams.val_percent)
            n_train = len(dataset) - n_val
            self.n_train = n_train
            self.n_val = n_val
            train, val = random_split(dataset, [n_train, n_val])

            # dataloader
            train_loader = train_dataloader(train, batch_size=self.hparams.batch)
            val_loader = train_dataloader(val, batch_size=self.hparams.batch, ar=True)
            train_all = train_dataloader(dataset, batch_size=len(dataset), ar=True)
            self.all_train = train_all
            i += 1
            i %= (len_dir-1)
            yield {
                'train': train_loader,
                'val': val_loader,
                'all': train_all,
            }

    def refresh_dataloader(self):
        # self._data = self.__dataloader().__next__()
        return next(self._generator)

    # @pl.data_loader
    def train_dataloader(self):
        return self._data['train']

    # @pl.data_loader
    def val_dataloader(self):
        return self._data['val']

    def all_dataloader(self):
        return self._data['all']

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    def scheduler(self):
        return optim.lr_scheduler.ReduceLROnPlateau(self.configure_optimizers(), 'min' if self.n_classes > 1 else 'max',
                                                    patience=2)


if __name__ == '__main__':
    hparams = get_opts()
    systems = TrainSystem(hparams)
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(f'ckpts/{hparams.exp_name}',
                                                                '{epoch:02d}'),
                                          monitor='Dice/test',
                                          mode='max',
                                          save_top_k=5, )

    logger = TestTubeLogger(
        save_dir="./logs",
        name=hparams.exp_name,
        debug=False,
        create_git_tag=False
    )

    if hparams.load_ckpt == 'model':
        trainer = Trainer(max_epochs=hparams.num_epochs,
                          checkpoint_callback=checkpoint_callback,
                          resume_from_checkpoint=hparams.ckpt_path,
                          logger=logger,
                          # early_stop_callback=None,
                          weights_summary=None,
                          progress_bar_refresh_rate=1,
                          gpus=hparams.num_gpus,
                          distributed_backend='ddp' if hparams.num_gpus > 1 else None,
                          num_sanity_val_steps=0 if hparams.num_gpus > 1 else 5,
                          # benchmark=True,
                          precision=16 if hparams.use_amp else 32,
                          amp_level='O2')
    else:
        trainer = Trainer(max_epochs=hparams.num_epochs,
                          checkpoint_callback=checkpoint_callback,
                          # resume_from_checkpoint=hparams.ckpt_path,
                          logger=logger,
                          # early_stop_callback=None,
                          weights_summary=None,
                          progress_bar_refresh_rate=1,
                          gpus=hparams.num_gpus,
                          distributed_backend='ddp' if hparams.num_gpus > 1 else None,
                          num_sanity_val_steps=0 if hparams.num_gpus > 1 else 5,
                          # benchmark=True,
                          precision=16 if hparams.use_amp else 32,
                          amp_level='O2')

    trainer.fit(systems)
