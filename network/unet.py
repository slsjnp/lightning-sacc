# U-Net网络


import torch.nn as nn
import torch
from torch import autograd


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        self.n_channels = 1
        self.n_classes = 1
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out


# ----------------------------------------

# # U-Net网络
#
#
# import torch.nn as nn
# import torch
# import numpy as np
# from PIL import Image
# from torch import autograd
# from torch.autograd import Variable
# from torchvision.transforms import transforms
#
#
# class DoubleConv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(DoubleConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, input):
#         return self.conv(input)
#
#
# class Down(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(Down, self).__init__()
#         self.downconv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_ch, out_ch)
#         )
#
#     def forward(self, x):
#         return self.downconv(x)
#
#
# class Up(nn.Module):
#     def __init__(self, in_ch, out_ch, transform=True):
#         super(Up, self).__init__()
#         if transform:
#             self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
#         else:
#             self.up = nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#                 nn.Conv2d(in_ch, in_ch // 2, 2, padding=0),
#                 nn.ReLU(inplace=True)
#             )
#         self.conv = DoubleConv(in_ch, out_ch)
#         # self.up.apply(self.init_weights)
#
#     def forward(self, x_down, x_left):
#         '''
#             input is BCHW
#             conv output shape = (input_shape - Filter_shape + 2 * padding)/stride + 1
#         '''
#         x1 = self.up(x_down)
#         x = torch.cat([x1, x_left], 1)
#         x2 = self.conv(x)
#         return x2
#
#     # def forward(self, x1, x2):
#     #     x1 = self.up(x1)
#     #     # input is CHW
#     #     diffY = x2.size()[2] - x1.size()[2]
#     #     diffX = x2.size()[3] - x1.size()[3]
#     #
#     #     x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#     #                     diffY // 2, diffY - diffY // 2])
#     #     # if you have padding issues, see
#     #     # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#     #     # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#     #     x = torch.cat([x2, x1], dim=1)
#     #     return self.conv(x)
#
#
# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# class Unet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(Unet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         # factor = 2 if bilinear else 1
#         factor = 1
#         self.down4 = Down(512, 1024 // factor)
#         self.up1 = Up(1024, 512 // factor)
#         self.up2 = Up(512, 256 // factor)
#         self.up3 = Up(256, 128 // factor)
#         self.up4 = Up(128, 64)
#         self.outc = OutConv(64, n_classes)
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits
#
#
# # class Unet(nn.Module):
# #     def __init__(self, in_ch, out_ch):
# #         super(Unet, self).__init__()
# #
# #         self.conv1 = DoubleConv(in_ch, 64)
# #         self.pool1 = nn.MaxPool2d(2)
# #         self.conv2 = DoubleConv(64, 128)
# #         self.pool2 = nn.MaxPool2d(2)
# #         self.conv3 = DoubleConv(128, 256)
# #         self.pool3 = nn.MaxPool2d(2)
# #         self.conv4 = DoubleConv(256, 512)
# #         self.pool4 = nn.MaxPool2d(2)
# #         self.conv5 = DoubleConv(512, 1024)
# #         self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
# #         self.conv6 = DoubleConv(1024, 512)
# #         self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
# #         self.conv7 = DoubleConv(512, 256)
# #         self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
# #         self.conv8 = DoubleConv(256, 128)
# #         self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
# #         self.conv9 = DoubleConv(128, 64)
# #         self.conv10 = nn.Conv2d(64, out_ch, 1)
# #
# #     def forward(self, x):
# #         c1 = self.conv1(x)
# #         p1 = self.pool1(c1)
# #         c2 = self.conv2(p1)
# #         p2 = self.pool2(c2)
# #         c3 = self.conv3(p2)
# #         p3 = self.pool3(c3)
# #         c4 = self.conv4(p3)
# #         p4 = self.pool4(c4)
# #         c5 = self.conv5(p4)
# #         up_6 = self.up6(c5)
# #         merge6 = torch.cat([up_6, c4], dim=1)
# #         c6 = self.conv6(merge6)
# #         up_7 = self.up7(c6)
# #         merge7 = torch.cat([up_7, c3], dim=1)
# #         c7 = self.conv7(merge7)
# #         up_8 = self.up8(c7)
# #         merge8 = torch.cat([up_8, c2], dim=1)
# #         c8 = self.conv8(merge8)
# #         up_9 = self.up9(c8)
# #         merge9 = torch.cat([up_9, c1], dim=1)
# #         c9 = self.conv9(merge9)
# #         c10 = self.conv10(c9)
# #         out = nn.Sigmoid()(c10)
# #
# #         return out
#
#
class FeatureExtractor(nn.Module):
    def __init__(self, submoudule, extracted_layers):
        """
        inspect feature map
        :param submoudule: Module
        :param extracted_layers: layer for inspect
        """
        super(FeatureExtractor, self).__init__()
        self.submodule = submoudule
        self.extracted_layers = extracted_layers
        self.dict = {}
        self.outputs = {}
        self.step = 0

    def forward(self, x, saved_path='/home/sj/workspace/jupyter/data/unet/feature_img'):
        tmplist = ['conv4', 'conv3', 'conv2', 'conv1']
        catlist = ['conv6', 'conv7', 'conv8', 'conv9']
        output_list = {}
        for name, module in self.submodule._modules.items():
            if name is "conv10":
                # x = x.view(x.size(0), -1)
                x = module(x)
                out = nn.Sigmoid()(x)
                return out
            if name in tmplist:
                x = module(x)
                self.dict[tmplist.index(name)] = x
            elif name in catlist:
                # a = x
                # b = self.dict[catlist.index(name)]
                merge6 = torch.cat([x, self.dict[catlist.index(name)]], dim=1)
                x = module(merge6)
            else:
                x = module(x)
            # x = module(x)
            if self.step == 50:
                if name in self.extracted_layers:
                    # save_name = 'epoch{}_{}'.format(epoch, name)
                    # output_list[name] = x
                    # self.outputs.append(output_list)
                    self.outputs[name] = x
        # y = module(x).cpu()
        # y = torch.squeeze(x)
        # y = y.data.numpy()
        # print(y.shape)
        # np.savetxt(saved_path, y, delimiter=',')
        return x
#
#
# def extractor(img_path, saved_path, net, use_gpu):
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor()]
#     )
#
#     img = Image.open(img_path)
#     img = transform(img)
#     print(img.shape)
#
#     x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
#     print(x.shape)
#
#     if use_gpu:
#         x = x.cuda()
#         net = net.cuda()
#     y = net(x).cpu()
#     y = torch.squeeze(y)
#     y = y.data.numpy()
#     print(y.shape)
#     np.savetxt(saved_path, y, delimiter=',')
#
#
# if __name__ == '__main__':
#     print('hello world')
