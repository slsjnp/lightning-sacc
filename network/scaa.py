import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    # def __init__(self, in_ch, out_ch, num_features, mid_ch=None):
    def __init__(self, conv, inplanes, planes, num_features, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, res=True):
        # self.conv3d = nn.Conv3d(in_ch, out_ch, ker)
        # self.ReLU1 = nn.ReLU(inplace=True)
        super(BasicBlock, self).__init__()
        self.res = res
        self.resconv = nn.Sequential(
            conv(inplanes, planes, kernel_size=3, padding=1, stride=stride, groups=groups),
            # nn.InstanceNorm3d(num_features),
            norm_layer(num_features),
            nn.ReLU(inplace=True),
            conv(planes, planes, kernel_size=3, padding=1),
            # nn.InstanceNorm3d(num_features),
            norm_layer(num_features),
            # nn.ReLU(inplace=True),
        )
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.res:
            x1 = x,
            x2 = self.conv3d(x)
            x3 = x1 + x2
            return self.ReLU(x3)
        else:
            x = self.resconv(x)
            x = self.ReLU(x)
            return x


class Encoder3d(nn.Module):
    def __init__(self, block, layers, conv, num_classes, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        """
        :param block: (required)
        :param layers: default [2, 2, 2, 2] (required)
        :param num_classes: (required)
        :param zero_init_residual:
        :param groups:
        :param width_per_group:
        :param replace_stride_with_dilation:
        :param norm_layer:  (required)
        """
        super(Encoder3d, self).__init__()
        self.num_classes = num_classes
        self.downsample1 = nn.MaxPool3d(2, stride=2)
        # self.resblock3d1 = ResBlock3d()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        self.inplanes = 1
        self.conv = conv
        # base block, planes, blocks
        self.layer1 = self._make_layer(block, 24, layers[0])
        x1 = self.layer1

        self.downsample2 = nn.MaxPool3d(2, stride=2)
        self.layer2 = self._make_layer(block, 32, layers[1])
        x2 = self.layer2

        self.downsample3 = nn.MaxPool3d(2, stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2])
        x3 = self.layer3

        self.downsample4 = nn.MaxPool3d(2, stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3])
        x4 = self.layer4

        # upsample
        self.upsample = nn.Upsample(scale_factor=16)
        self.final_conv = nn.Conv3d(in_channels=64, out_channels=num_classes, kernel_size=1)
        x5 = self.final_conv

    def _make_layer(self, block, planes, blocks=2, stride=1, dilate=False):
        downsample = None
        norm_layer = self._norm_layer
        layers = []
        layers.append(block(self.conv, self.inplanes, planes, self.num_classes, stride, downsample, self.groups,
                            self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.conv, self.inplanes, planes, self.num_classes, stride, downsample, groups=self.groups,
                      base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.downsample1(x)
        x = self.layer1(x)
        f3d2 = x

        x = self.downsample2(x)
        x = self.layer2(x)
        f3d3 = x

        x = self.downsample3(x)
        x = self.layer3(x)
        f3d4 = x

        x = self.downsample4(x)
        x = self.layer4(x)
        f3d5 = x

        x = self.upsample(x)
        x = self.final_conv(x)
        f3d6 = x

        return f3d2, f3d3, f3d4, f3d5, f3d6


class ResBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch, num_features):
        # self.conv3d = nn.Conv3d(in_ch, out_ch, ker)
        # self.ReLU1 = nn.ReLU(inplace=True)
        super(ResBlock3d, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(num_features),
            # nn.ReLU(inplace=True),
        )
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = x,
        x2 = self.conv3d(x)
        x3 = x1 + x2
        return self.ReLU(x3)


class ConBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch, num_features):
        super(ConBlock2d, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        # self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv2d(x)
        return x


# 24 ，2
class MSFA(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, shape=4, dropout=0.0):
        super(MSFA, self).__init__()
        self.wq = nn.Conv3d(in_ch, mid_ch, 1)
        self.wk = nn.Conv2d(in_ch, mid_ch, 1)
        self.wm = nn.Conv2d(in_ch, mid_ch, 1)
        self.adapt_pool_3d = nn.AdaptiveMaxPool3d((None, shape, shape))
        self.adapt_pool_2d = nn.AdaptiveMaxPool2d(shape)
        self.dot_product_attention = ScaledDotProductAttention(dropout)

    def forward(self, f2d, f3d, shape):
        x1 = f3d,
        x2 = self.wk(f3d),
        # x3 = nn.AdaptiveMaxPool3d(shape)(x2)
        x3 = self.adapt_pool_3d(x2)
        x4 = torch.flatten(x3, start_dim=1)

        y1 = f2d,
        y2 = self.wq(f2d),
        y3 = nn.AdaptiveMaxPool2d(shape)(y2)
        y4 = torch.flatten(y3, start_dim=1)

        z1 = self.dot_product_attention(y4, x4, x1)  # scale=0.125


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        assert model_dim % num_heads == 0
        # ?
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        # self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        # self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        # self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        # self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        # resdiual
        residual = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = query.size(0)

        # # Linear projection
        # key = self.linear_k(key)
        # value = self.linear_v(value)
        # query = self.linear_q(query)

        K = key.view(batch_size, -1, num_heads, dim_per_head).permute(0, 2, 1, 3)
        Q = query.view(batch_size, -1, num_heads, dim_per_head).permute(0, 2, 1, 3)
        V = value.view(batch_size, -1, num_heads, dim_per_head).permute(0, 2, 1, 3)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        # scaled dot product attention
        scale = (key.size(-1)) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention

    # def forward(self, key, value, query, attn_mask=None):
    #     # resdiual
    #     residual = query
    #     dim_per_head = self.dim_per_head
    #     num_heads = self.num_heads
    #     batch_size = query.size(0)
    #
    #     # Linear projection
    #     key = self.linear_k(key)
    #     value = self.linear_v(value)
    #     query = self.linear_q(query)
    #
    #     # split by heads
    #     key = key.view(batch_size * num_heads, -1, dim_per_head)
    #     value = value.view(batch_size * num_heads, -1, dim_per_head)
    #     query = query.view(batch_size * num_heads, -1, dim_per_head)
    #
    #     if attn_mask:
    #         attn_mask = attn_mask.repeat(num_heads, 1, 1)
    #
    #     # scaled dot product attention
    #     scale = (key.size(-1)) ** -0.5
    #     context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)
    #
    #     # concat heads
    #     context = context.view(batch_size, -1, dim_per_head * num_heads)
    #
    #     # final linear projection
    #     output = self.linear_final(context)
    #
    #     # dropout
    #     output = self.dropout(output)
    #
    #     # add residual and norm layer
    #     output = self.layer_norm(residual + output)
    #
    #     return output, attention


class SCAA(nn.Module):
    def __init__(self):
        super(SCAA, self).__init__()
        # 3d encoder
        self.encoder3d = Encoder3d()

        # 2d encoder layer 1
        # self.encoder2d1 = ConBlock2d()
        # self.download1 = Download()
        # self.msfa1 = MSFA()
        self.down_layer1 = self._make_layer()
        self.down_layer2 = self._make_layer()
        self.down_layer3 = self._make_layer()
        self.down_layer4 = self._make_layer()

        # 2d decoder
        self.up1 = 



    def _make_layer(self):
        pass

    # forward
    def forward(self):
        # 3d encoder => output 4 feature maps

        # 3d decoder => output for training only
        # 2d encoder => output 4 feature maps for residual
        # 3d + 2d => MSFA

        # 2d decoder

        # concat

        # task encoder

        # dynamic head

        # layer for output

        pass
