import logging
import re

import numpy as np
import os
from glob import glob

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

osp = os.path
join = os.path.join
listdir = os.listdir
splitext = os.path.splitext


class BaseDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, transform, target_transform, scale=1, mask_suffix='', Base=True):
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.target_transform = target_transform
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        if Base == True:
            self.imgs_dir = imgs_dir
            self.masks_dir = masks_dir
            self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        logging.info(f'Create dataset with len(self.ids) example')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newH > 0 and newW > 0, f'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
        img_nd = np.array(pil_img)
        # ensure img_nd is three dim as (h, w, c)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        # transpose to c, h ,w
        # img_trans = img_nd.transpose((2, 0, 1))
        img_trans = img_nd
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans.astype(np.float64)

    def __getitem__(self, i):
        idx = self.ids[i]
        # mask_name = self.masks_dir + idx + self.mask_suffix + '.*'
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        # ensure only one file
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {idx}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        # img.convert('RGB')
        assert img.size == mask.size, f'Image and mask {idx} should be the same size'
        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        # return self.transform(img).float(), self.target_transform(mask).float()

        return {
            # 'image': torch.from_numpy(img).type(torch.FloatTensor),
            # 'mask': torch.from_numpy(mask).type(torch.FloatTensor)
            'image': self.transform(img).float(),
            'mask': self.target_transform(mask).float()
        }


def get_data_function(func):
    def printer():
        func()

    return printer


class ChaoDataset(BaseDataset):
    def __init__(self, imgs_dir, masks_dir, transform, target_transform, index, scale=1, mask_suffix='', type='CT'):
        super(ChaoDataset, self).__init__(imgs_dir, masks_dir, transform, target_transform, scale, mask_suffix,
                                          Base=False)
        self.type = type
        self.imgs_dir_root = imgs_dir
        self.masks_dir_root = masks_dir
        self.sum_ids = []
        self.sum_mask_ids = []
        if type == 'CT':
            path = self.imgs_dir_root + 'CT/'
            # for case in os.listdir(path):
            case = index
            if case is not None:
                dcm_data = path + case + '/CTdcm2png/'
                label_data = path + case + '/Ground/'
                # diff_part = 'CT/' + case
                dcm_list = listdir(dcm_data)
                re_pattern = re.compile(r'(\d+)')
                label_list = listdir(label_data)
                dcm_list.sort(key=lambda x: re_pattern.findall(x)[-1] + re_pattern.findall(x)[0])
                label_list.sort(key=lambda x: re_pattern.findall(x)[-1])
                # ids = []
                # mask_ids = []
                for file in dcm_list:
                    result = dcm_data + file[:-4]
                    # ids.append(result)

                    # label_result = label_data + 'liver_GT_' + file[-7:-4]
                    label_result = label_data + label_list[dcm_list.index(file)][:-4]
                    # mask_ids.append(label_result)
                    self.sum_ids.append(result)
                    self.sum_mask_ids.append(label_result)
                # ids = [dcm_data + splitext(file)[0] for file in listdir(dcm_data) if
                #        not file.startswith('.')]
                # mask_ids = [label_data + splitext(file)[0] for file in listdir(label_data) if
                #             not file.startswith('.')]
                # self.ids.extend(ids)
                # self.mask_ids.extend(mask_ids)

                # save_path = path + case + '/CTdcm2png/'
                # for img in os.listdir(dcm_data):
                # multi_dcm(dcm_data, save_path)
        # 未改
        elif type == 'MR':
            path = self.imgs_dir_root + 'MR/'
            for case in os.listdir(path):
                dcm_data = path + case + '/T2SPIR/MRdcm2png/'
                label_data = path + case + '/T2SPIR/Ground/'
                ids = [dcm_data + splitext(file)[0] for file in listdir(dcm_data) if not file.startswith('.')]
                mask_ids = [label_data + splitext(file)[0] for file in listdir(label_data) if
                            not file.startswith('.')]
                self.ids.extend(ids)
                self.mask_ids.extend(mask_ids)
                # save_path = path + case + '/T2SPIR/MRdcm2png/'
                # for img in os.listdir(dcm_data):
                # multi_dcm(dcm_data, save_path)
        self.files = []

        spacing = {
            0: [0.8, 0.8, 1.5],
            1: [0.8, 0.8, 1.5],
            2: [0.8, 0.8, 1.5],
            3: [0.8, 0.8, 1.5],
            4: [0.8, 0.8, 1.5],
            5: [0.8, 0.8, 1.5],
            6: [0.8, 0.8, 1.5],
        }
        print("Start preprocessing....")
        for item in self.sum_ids:
            print(item)
            # image_path, label_path = item
            # task_id = int(image_path[16 + 5])
            # if task_id != 1:
            #     name = osp.splitext(osp.basename(label_path))[0]
            # else:
            #     name = label_path[31 + 5:41 + 5]
            # img_file = osp.join(self.root, image_path)
            # label_file = osp.join(self.root, label_path)
            # label = nib.load(label_file).get_data()
            task_id = 1
            # if task_id == 1:
            # label = label.transpose((1, 2, 0))
            # boud_h, boud_w, boud_d = np.where(label >= 1)
            self.files.append({
                # "image": img_file,
                # "label": label_file,
                # "name": name,
                "task_id": task_id,
                "spacing": spacing[task_id],
                # "bbx": [boud_h, boud_w, boud_d]
            })
        # print('{} images are loaded!'.format(len(self.img_ids)))

    @classmethod
    def get_list(cls, imgs_dir, masks_dir, scale=1, mask_suffix='', type='CT'):
        type = type
        imgs_dir_root = imgs_dir
        masks_dir_root = masks_dir
        if type == 'CT':
            path = imgs_dir_root + 'CT/'
            return os.listdir(path)
            # for case in os.listdir(path):

            # case = '22'
            # if case is not None:

    def __len__(self):
        # sums = 0
        # for i in self.sum_ids:
        #     sums += len(i)
        sums = len(self.sum_ids)
        return sums

    # @get_data_function(i)
    # def __getitem__(self, i):
    #     # img = self.sum_ids[i]
    #     # mask = self.sum_mask_ids[i]
    #     # return self._getitem(img, mask, i)
    #     return self._getitem(self.sum_ids, self.sum_mask_ids)

    def __getitem__(self, i):
        idx = self.sum_ids[i]
        mask_idx = self.sum_mask_ids[i]
        # mask_name = self.masks_dir + idx + self.mask_suffix + '.*'
        a = mask_idx + self.mask_suffix + '.*'
        mask_file = glob(mask_idx + self.mask_suffix + '.*')
        img_file = glob(idx + '.*')
        # ensure only one file
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {idx}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        # mask.show()
        # img.show()
        # img.convert('RGB')
        assert img.size == mask.size, f'Image and mask {idx} should be the same size'
        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        # return self.transform(img).float(), self.target_transform(mask).float()

        return {
            # 'image': torch.from_numpy(img).type(torch.FloatTensor),
            # 'mask': torch.from_numpy(mask).type(torch.FloatTensor)
            'image': self.transform(img).float(),
            'mask': self.target_transform(mask).float(),
            'files': self.files
        }


class WholeDataset(Dataset):

    def __init__(self, imgs_dir, masks_dir, transform, target_transform, scale=1, mask_suffix='', Base=True):
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.target_transform = target_transform
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        if Base == True:
            self.imgs_dir = imgs_dir
            self.masks_dir = masks_dir
            self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        logging.info(f'Create dataset with len(self.ids) example')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newH > 0 and newW > 0, f'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
        img_nd = np.array(pil_img)
        # ensure img_nd is three dim as (h, w, c)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        # transpose to c, h ,w
        # img_trans = img_nd.transpose((2, 0, 1))
        img_trans = img_nd
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans.astype(np.float64)

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_name = self.masks_dir + 'segmentation' + idx[6:] + self.mask_suffix + '.*'
        # mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')

        mask_file = glob(mask_name)
        img_file = glob(self.imgs_dir + idx + '.*')
        # ensure only one file
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {idx}'
        # mask = Image.open(mask_file[0])
        # img = Image.open(img_file[0])
        img = nib.load(img_file[0])
        mask = nib.load(mask_file[0])
        width, height, queue = img.dataobj.shape
        img = img.get_fdata()
        mask = mask.get_fdata()

        # img.convert('RGB')
        assert img.size == mask.size, f'Image and mask {idx} should be the same size'
        if img.max() > 1:
            img = img / 255
        # img = self.preprocess(img, self.scale)
        # mask = self.preprocess(mask, self.scale)
        # return self.transform(img).float(), self.target_transform(mask).float()

        return {
            # 'image': torch.from_numpy(img).type(torch.FloatTensor),
            # 'mask': torch.from_numpy(mask).type(torch.FloatTensor)
            'image': self.transform(img).float(),
            'mask': self.target_transform(mask).float(),

        }


def train_dataloader(train_data_list, batch_size, ar=None):
    """

    :param ar:
    :param train_data_list:
    :param batch_size:
    :return: Dataloader
    """

    # if ar is None:
    #     result = DataLoader(dataset=train_data_list, batch_size=batch_size, shuffle=True, num_workers=16,
    #                         pin_memory=True)
    # else:
    #     result = DataLoader(dataset=train_data_list, batch_size=batch_size, shuffle=False, num_workers=16,
    #                         pin_memory=True, drop_last=True)
    # return result

    if ar is None:
        result = DataLoader(dataset=train_data_list, batch_size=batch_size, shuffle=True, num_workers=4,
                            pin_memory=True)
    else:
        result = DataLoader(dataset=train_data_list, batch_size=batch_size, shuffle=False, num_workers=4,
                            pin_memory=True, drop_last=True)
    return result
