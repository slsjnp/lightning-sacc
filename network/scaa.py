import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    # def __init__(self, in_ch, out_ch, num_features, mid_ch=None):

    # def __init__(self, conv, inplanes, planes, num_features, groups=1, stride=1, downsample=None,
    #              base_width=64, norm_layer=None, res=True):
    #     # self.conv3d = nn.Conv3d(in_ch, out_ch, ker)
    #     # self.ReLU1 = nn.ReLU(inplace=True)
    #     super(BasicBlock, self).__init__()
    #     self.res = res
    #     self.conv = conv(inplanes, planes, kernel_size=1, stride=stride, groups=groups)
    #
    #     self.resconv = nn.Sequential(
    #         conv(inplanes, planes, kernel_size=3, padding=1, stride=stride, groups=groups),
    #         # nn.InstanceNorm3d(num_features),
    #         norm_layer(planes),
    #         nn.ReLU(inplace=True),
    #
    #         conv(planes, planes, kernel_size=3, padding=1),
    #         # nn.InstanceNorm3d(num_features),
    #         norm_layer(planes),
    #         # nn.ReLU(inplace=True),
    #     )
    #     self.ReLU = nn.ReLU(inplace=True)

    def __init__(self, conv, inplanes, planes, num_features, groups=1, stride=1, downsample=None,
                 base_width=64, norm_layer=None, res=True):
        # self.conv3d = nn.Conv3d(in_ch, out_ch, ker)
        # self.ReLU1 = nn.ReLU(inplace=True)
        super(BasicBlock, self).__init__()
        self.res = res
        self.conv = conv(inplanes, planes, kernel_size=1, stride=stride, groups=groups)
        mid_ch = 1
        if inplanes / 2 >= 1:
            mid_ch = int(inplanes / 2)
        # mid_ch
        self.resconv = nn.Sequential(
            conv(inplanes, mid_ch, kernel_size=3, padding=1, stride=stride, groups=groups),
            conv(mid_ch, mid_ch, kernel_size=3, padding=1, stride=stride, groups=groups),
            # nn.InstanceNorm3d(num_features),
            norm_layer(mid_ch),
            nn.ReLU(inplace=True),
            conv(mid_ch, planes, kernel_size=3, padding=1),
            # nn.InstanceNorm3d(num_features),
            norm_layer(planes),
            # nn.ReLU(inplace=True),
        )
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.res:
            x1 = self.conv(x)
            # x1 = x
            x2 = self.resconv(x)
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
        Encoder3d()
        整张CT图作为输入
        :param block: (required) BasicBlock(nn.Conv3d, 1, 24, num_features=2)
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
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        self.dilation = 1
        self.inplanes = 1
        self.conv = conv
        # base block, planes, blocks
        self.layer1 = self._make_layer(block, 24, layers[0], stride=1, norm_layer=self._norm_layer)
        x1 = self.layer1

        self.downsample2 = nn.MaxPool3d(2, stride=2)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=1, norm_layer=self._norm_layer)
        x2 = self.layer2

        self.downsample3 = nn.MaxPool3d(2, stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=1, norm_layer=self._norm_layer)
        x3 = self.layer3

        self.downsample4 = nn.MaxPool3d(2, stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=1, norm_layer=self._norm_layer)
        x4 = self.layer4

        # upsample
        self.upsample = nn.Upsample(scale_factor=16)
        self.final_conv = nn.Conv3d(in_channels=64, out_channels=num_classes, kernel_size=1)
        x5 = self.final_conv

    def _make_layer(self, block, planes, blocks=2, stride=1, dilate=False, norm_layer=None):
        downsample = None
        layers = []
        layers.append(
            block(self.conv, self.inplanes, planes, self.num_classes, stride=stride, downsample=downsample,
                  norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.conv, self.inplanes, planes, self.num_classes, stride=stride, downsample=downsample,
                      norm_layer=norm_layer))

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


# class ResBlock3d(nn.Module):
#     def __init__(self, in_ch, out_ch, mid_ch, num_features):
#         # self.conv3d = nn.Conv3d(in_ch, out_ch, ker)
#         # self.ReLU1 = nn.ReLU(inplace=True)
#         super(ResBlock3d, self).__init__()
#         self.conv3d = nn.Sequential(
#             nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1),
#             nn.InstanceNorm3d(num_features),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1),
#             nn.InstanceNorm3d(num_features),
#             # nn.ReLU(inplace=True),
#         )
#         self.ReLU = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x1 = x,
#         x2 = self.conv3d(x)
#         x3 = x1 + x2
#         return self.ReLU(x3)


class ConBlock2d(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, num_features):
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
    def __init__(self, in_ch2, in_ch3, mid_ch, shape, dropout=0.0):
        super(MSFA, self).__init__()
        self.wq = nn.Conv2d(in_ch2, mid_ch, 1)
        self.wk = nn.Conv3d(in_ch3, mid_ch, 1)
        self.wm = nn.Conv2d(mid_ch, 1, 1)
        self.adapt_pool_3d = nn.AdaptiveMaxPool3d((None, shape, shape))
        self.adapt_pool_2d = nn.AdaptiveMaxPool2d(shape)
        self.dot_product_attention = ScaledDotProductAttention(mid_ch, dropout)

    def forward(self, f2d, f3d):
        x1 = f3d
        x2 = self.wk(f3d)
        # x3 = nn.AdaptiveMaxPool3d(shape)(x2)
        x3 = self.adapt_pool_3d(x2)
        x4 = torch.flatten(x3, start_dim=3)

        y1 = f2d
        y2 = self.wq(f2d)
        y3 = self.adapt_pool_2d(y2)
        y4 = torch.flatten(y3, start_dim=2)
        # q, k, v
        z1, attention = self.dot_product_attention(y4, x4, x1)  # scale=0.125
        x = self.wm(z1)
        x = torch.cat([x, f2d], dim=1)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, channel, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=3)
        self.channel = channel
        self.conv = None
        self.conv2 = None

    def forward(self, q, k, v, scale=None, attn_mask=None):
        # q.unsqueeze(3)
        # b c d 1
        attention = torch.matmul(q.unsqueeze(3).transpose(2, 3), k.squeeze().transpose(1, 2)).transpose(2, 3)
        # attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        bv, cv, dv, hv, wv = v.shape
        ba, ca, da, na = attention.shape
        self.conv = nn.Conv2d(cv, 1, 1).cuda()
        self.conv2 = nn.Conv2d(da, 1, 1).cuda()
        att = attention
        attention = attention.expand(ba, ca, da, hv * wv)
        v = torch.flatten(v, start_dim=3)
        result = []

        for i in range(self.channel):
            x = attention[:, i, :, :].unsqueeze(dim=1)
            tmp = torch.mul(x, v)
            tmp = self.conv(tmp).squeeze().view(ba, da, hv, wv)
            # tmp = self.conv2(tmp).squeeze()
            tmp = self.conv2(tmp)
            result.append(tmp)
        feature = result[0]
        for index in range(1, len(result)):
            feature = torch.cat((feature, result[index]), dim=1)
            a = 1
        return feature, att


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


class SCAA3D(nn.Module):
    def __init__(self, num_features=2):
        super(SCAA3D, self).__init__()
        self.encoder3d = Encoder3d(BasicBlock, layers=[2, 2, 2, 2], conv=nn.Conv3d, num_classes=num_features)

    def forward(self, x):
        f3d2, f3d3, f3d4, f3d5, f3d6 = self.encoder3d(x)
        return f3d2, f3d3, f3d4, f3d5, f3d6


class SCAA(nn.Module):
    def __init__(self, num_features=2):
        super(SCAA, self).__init__()
        # 3d encoder

        self.n_classes = num_features
        # 2d encoder layer 1
        # self.encoder2d1 = ConBlock2d()
        # self.download1 = Download()
        self.first_conv = ConBlock2d(1, 64, 64, num_features=num_features)

        self.down_layer1 = self._make_layer()
        self.msfa1 = MSFA(64, 24, 2, 16)
        self.conv1 = ConBlock2d(65, 96, 96, num_features=num_features)

        self.down_layer2 = self._make_layer()
        self.msfa2 = MSFA(96, 32, 2, 8)
        self.conv2 = ConBlock2d(97, 128, 128, num_features=num_features)

        self.down_layer3 = self._make_layer()
        self.msfa3 = MSFA(128, 64, 4, 4)
        self.conv3 = ConBlock2d(129, 192, 192, num_features=num_features)

        self.down_layer4 = self._make_layer()
        self.msfa4 = MSFA(192, 64, 4, 4)
        self.conv4 = ConBlock2d(193, 256, 256, num_features=num_features)

        # 2d decoder
        self.up1 = Up(256, 320, 128, num_features=num_features)
        self.up2 = Up(128, 192, 64, num_features=num_features)
        self.up3 = Up(64, 128, 32, num_features=num_features)
        self.up4 = Up(32, 80, 16, num_features=num_features)

        # 分类数量为最后的通道数
        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_features, kernel_size=1)
        )
        self.GAP = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        # 拼接后要 +分类数量
        self.controller = nn.Conv2d(256 + 2, 162, kernel_size=1, stride=1, padding=0)

    def encoding_task(self, task_id):
        N = task_id.shape[0]
        # N = 1
        task_encoding = torch.zeros(size=(N, 2))
        for i in range(N):
            task_encoding[i, task_id[i]] = 1
        return task_encoding.cuda()

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 2, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 2)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def _make_layer(self):
        return nn.AvgPool2d(2, stride=2)

    # forward
    # def forward(self, x, f3d2, f3d3, f3d4, f3d5, f3d6, task_id):
    def forward(self, x, feature, task_id):
        (f3d2, f3d3, f3d4, f3d5, f3d6) = feature

        # 3d encoder => output 4 feature maps
        # f3d2, f3d3, f3d4, f3d5, f3d6 = self.encoder3d(x)

        # 3d decoder => output for training only

        # 2d encoder => output 4 feature maps for residual
        ################################################################################################################
        # encoder
        ################################################################################################################

        x = self.first_conv(x)
        slip1 = x

        x = self.down_layer1(x)
        x = self.msfa1(x, f3d2)
        x = self.conv1(x)
        slip2 = x

        x = self.down_layer2(x)
        x = self.msfa2(x, f3d3)
        x = self.conv2(x)
        slip3 = x

        x = self.down_layer3(x)
        x = self.msfa3(x, f3d4)
        x = self.conv3(x)
        slip4 = x

        x = self.down_layer4(x)
        x = self.msfa4(x, f3d5)
        x = self.conv4(x)

        x_feat = self.GAP(x)
        ################################################################################################################
        # decoder (conv + upsample)
        ################################################################################################################
        x = self.up1(slip4, x)
        x = self.up2(slip3, x)
        x = self.up3(slip2, x)
        x = self.up4(slip1, x)

        task_encoding = self.encoding_task(task_id)
        task_encoding.unsqueeze_(2).unsqueeze_(2)
        x_cond = torch.cat([x_feat, task_encoding], 1)
        params = self.controller(x_cond)
        params.squeeze_(-1).squeeze_(-1).squeeze_(-1)

        head_inputs = self.precls_conv(x)

        N, _, H, W = head_inputs.size()
        head_inputs = head_inputs.reshape(1, -1, H, W)

        weight_nums, bias_nums = [], []
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 2)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(2)
        weights, biases = self.parse_dynamic_params(params, self.n_classes, weight_nums, bias_nums)

        logits = self.heads_forward(head_inputs, weights, biases, N)

        logits = logits.reshape(-1, 2, H, W)

        # 3d + 2d => MSFA

        # 2d decoder

        # concat

        # task encoder

        # dynamic head

        # layer for output
        return logits


class Up(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, num_features):
        super(Up, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_ch, int(in_ch / 2), 2, stride=2)
        self.conv = ConBlock2d(mid_ch, out_ch, out_ch, num_features=num_features)

    def forward(self, slip, x):
        x = self.up(x)
        x = torch.cat([slip, x], dim=1)
        x = self.conv(x)
        return x
