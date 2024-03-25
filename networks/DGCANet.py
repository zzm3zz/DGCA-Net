import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from thop import profile


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=1, activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class DoubleConv(nn.Module):

    def __init__(self, cin, cout):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, padding=1),
            ConvBNReLU(cout, cout, 3, stride=1, padding=1, activation=False)
        )
        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.bn(x)
        x = h + x
        x = self.relu(x)
        return x


class DWCONV(nn.Module):
    """
    Depthwise Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=None):
        super(DWCONV, self).__init__()
        if groups == None:
            groups = in_channels
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=groups, bias=True
                                   )

    def forward(self, x):
        result = self.depthwise(x)
        return result


class UEncoder(nn.Module):

    def __init__(self):
        super(UEncoder, self).__init__()
        self.res1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.res2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.res3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.res4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.res5 = DoubleConv(256, 512)
        self.pool5 = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        x = self.res1(x)
        features.append(x)  # (224, 224, 64)
        x = self.pool1(x)  # (112, 112, 64)

        x = self.res2(x)
        features.append(x)  # (112, 112, 128)
        x = self.pool2(x)  # (56, 56, 128)

        x = self.res3(x)
        features.append(x)  # (56, 56, 256)
        x = self.pool3(x)  # (28, 28, 256)

        x = self.res4(x)
        features.append(x)  # (28, 28, 512)
        x = self.pool4(x)  # (14, 14, 512)

        x = self.res5(x)
        features.append(x)  # (14, 14, 1024)
        x = self.pool5(x)  # (7, 7, 1024)
        features.append(x)
        return features


class Dual_axis(nn.Module):

    def __init__(self, input_size, channels, d_h, d_v, d_w, heads, dropout, stage=0):
        super(Dual_axis, self).__init__()
        self.dwconv_qh = DWCONV(channels, channels)
        self.dwconv_kh = DWCONV(channels, channels)
        self.pool_qh = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_kh = nn.AdaptiveAvgPool2d((None, 1))
        self.fc_qh = nn.Linear(channels, heads * d_h)
        self.fc_kh = nn.Linear(channels, heads * d_h)
        self.stage = stage
        self.dwconv_v = DWCONV(channels, channels)
        self.fc_v = nn.Linear(channels, heads * d_v)

        self.dwconv_qw = DWCONV(channels, channels)
        self.dwconv_kw = DWCONV(channels, channels)
        self.pool_qw = nn.AdaptiveAvgPool2d((1, None))
        self.pool_kw = nn.AdaptiveAvgPool2d((1, None))
        self.fc_qw = nn.Linear(channels, heads * d_w)
        self.fc_kw = nn.Linear(channels, heads * d_w)

        self.fc_o = nn.Linear(heads * d_v, channels)

        self.channels = channels
        self.d_h = d_h
        self.d_v = d_v
        self.d_w = d_w
        self.heads = heads
        self.dropout = dropout
        self.attn_drop = nn.Dropout(0.1)
        self.scaled_factor_h = self.d_h ** -0.5
        self.scaled_factor_w = self.d_w ** -0.5
        self.Bh = nn.Parameter(torch.Tensor(1, self.heads, input_size, input_size), requires_grad=True)
        self.Bw = nn.Parameter(torch.Tensor(1, self.heads, input_size, input_size), requires_grad=True)

    def forward(self, q, k, v):
        global result
        b, c, h, w = q.shape

        # Get qh, kh, v, qw, kw
        qh = self.dwconv_qh(q)  # [b, c, h, w]
        # qh = x
        qh = self.pool_qh(qh).squeeze(-1).permute(0, 2, 1)  # [b, h, c]
        qh = self.fc_qh(qh)  # [b, h, heads*d_h]
        qh = qh.view(b, h, self.heads, self.d_h).permute(0, 2, 1,
                                                         3).contiguous()  # [b, heads, h, d_h] -> [3, 2, 112, 23]

        kh = self.dwconv_kh(k)  # [b, c, h, w]
        # kh = x
        kh = self.pool_kh(kh).squeeze(-1).permute(0, 2, 1)  # [b, h, c]
        kh = self.fc_kh(kh)  # [b, heads*d_h, h]
        kh = kh.view(b, h, self.heads, self.d_h).permute(0, 2, 1,
                                                         3).contiguous()  # [b, heads, h, d_h] -> [3, 2, 112, 23]

        v = self.dwconv_v(v)
        # v = x
        v_b, v_c, v_h, v_w = v.shape
        v = v.view(v_b, v_c, v_h * v_w).permute(0, 2, 1).contiguous()
        v = self.fc_v(v)
        v = v.view(v_b, v_h, v_w, self.heads, self.d_v).permute(0, 3, 1, 2, 4).contiguous()
        v = v.view(v_b, self.heads, v_h, v_w * self.d_v).contiguous()  # [b, heads, h, w*d_v]  -> [3, 2, 112, 1288]

        qw = self.dwconv_qw(q)  # [b, c, h, w]
        # qw = x
        qw = self.pool_qw(qw).squeeze(-2).permute(0, 2, 1)  # [b, w, c]
        qw = self.fc_qw(qw)  # [b, heads*d_w, w]
        qw = qw.view(b, w, self.heads, self.d_w).permute(0, 2, 1,
                                                         3).contiguous()  # [b, heads, w, d_w]  -> [3, 2, 56, 23]

        kw = self.dwconv_kw(k)  # [b, c, h, w]
        # kw = x
        kw = self.pool_kw(kw).squeeze(-2).permute(0, 2, 1)  # [b, w, c]
        kw = self.fc_kw(kw)  # [b, heads*d_w, w]
        kw = kw.view(b, w, self.heads, self.d_w).permute(0, 2, 1,
                                                         3).contiguous()  # [b, heads, w, d_w] -> [3, 2, 56, 23]

        if self.stage % 2 == 0:
            qh = qh.permute(0, 1, 3, 2)
            kh = kh.permute(0, 1, 3, 2)
            qh = F.normalize(qh, dim=3)
            kh = F.normalize(kh, dim=3)

            attn_h = qh @ kh.transpose(2, 3)  # b, heads, c, c
            attn_h = attn_h.softmax(dim=-1)
            attn_h = self.attn_drop(attn_h)

            qw = qw.permute(0, 1, 3, 2)
            kw = kw.permute(0, 1, 3, 2)
            qw = F.normalize(qw, dim=3)
            kw = F.normalize(kw, dim=3)
            attn_w = qw @ kw.transpose(2, 3)  # b, heads, c, c
            attn_w = attn_w.softmax(dim=-1)
            attn_w = self.attn_drop(attn_w)

            # v.shape = b, heads, h*w, c
            v = v.view(v_b, self.heads, v_h * v_w, self.d_v).permute(0, 1, 3, 2)  # b, heads, c, h*w

            # Attention
            result = (attn_h @ v).permute(0, 1, 3, 2)  # b, heads, h*w, c
            result = (result @ attn_w).permute(0, 2, 1, 3)  # b, h*w, heads, c
            # print(result.shape)
            shape = result.shape
            result = result.reshape(shape[0], shape[1], shape[2] * shape[3])

        else:
            # v.shape = b, heads, h*w, c
            h_v = v

            v = v.view(v_b, self.heads, v_h * v_w, self.d_v).permute(0, 1, 3, 2)  # b, heads, c, h*w
            w_v = v.view(v_b, self.heads, self.d_v, v_h, v_w).contiguous()  # b, heads, c*h, w
            w_v = w_v.view(v_b, self.heads, self.d_v * v_h, v_w).contiguous()  # b, heads, c*h, w
            w_v = w_v.permute(0, 1, 3, 2).contiguous()  # b, heads, w, c*h

            kh = kh.permute(0, 1, 3, 2).contiguous()  # b head c h
            qh = F.softmax(qh, dim=3)  # b head h c
            kh = F.softmax(kh, dim=2)
            attn_h = kh @ h_v  # b, heads, c, w*c

            kw = kw.permute(0, 1, 3, 2).contiguous()  # b head c w
            qw = F.softmax(qw, dim=3)
            kw = F.softmax(kw, dim=2)
            attn_w = kw @ w_v  # b, heads, c, c*h

            # Attention
            result_h = qh @ attn_h  # b, heads, h, w*c
            result_w = qw @ attn_w  # b, heads, w, c*h
            result_h = result_h.view(v_b, self.heads, v_h * v_w, self.d_v).contiguous()  # b, heads, h*w, c
            result_w = result_w.permute(0, 1, 3, 2)  # b, heads, c*h, w
            result_w = result_w.view(v_b, self.heads, self.d_v, v_h, v_w).contiguous()  # b, heads, c, h, w
            result_w = result_w.view(v_b, self.heads, self.d_v, v_h * v_w).contiguous()  # b, heads, c, h*w
            result_w = result_w.permute(0, 1, 3, 2).contiguous()  # b, heads, w*h, c
            result = result_h + result_w  # b, heads, w*h, c
            result = result.permute(0, 2, 1, 3).contiguous()  # b, n, heads, c
            result = result.view(b, h * w, self.heads * self.d_v).contiguous()

        result = self.fc_o(result).view(b, self.channels, h, w)  # [b, channels, h, w]
        return result


class FFN_MultiLN(nn.Module):
    def __init__(self, in_channels, img_size, R, drop=0.):
        super(FFN_MultiLN, self).__init__()
        exp_channels = in_channels * R
        self.h = img_size
        self.w = img_size
        self.fc1 = nn.Linear(in_channels, exp_channels)
        self.dwconv = nn.Sequential(
            DWCONV(exp_channels, exp_channels)
        )
        self.ln1 = nn.LayerNorm(exp_channels, eps=1e-6)
        self.ln2 = nn.LayerNorm(exp_channels, eps=1e-6)
        self.ln3 = nn.LayerNorm(exp_channels, eps=1e-6)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(exp_channels, in_channels)

    def forward(self, x):
        x = self.fc1(x)
        b, n, c = x.shape  # [b, hw, c]
        h = x

        x = x.view(b, self.h, self.w, c).permute(0, 3, 1, 2)  # [b, c, h, w]
        x = self.dwconv(x).view(b, c, self.h * self.w).permute(0, 2, 1)
        x = self.ln1(x + h)
        x = self.ln2(x + h)
        x = self.ln3(x + h)
        x = self.act1(x)

        x = self.fc2(x)
        return x


class IntraTransBlock(nn.Module):
    def __init__(self, img_size, stride, d_h, d_v, d_w, num_heads, R=4, in_channels=46, stage=0):
        super(IntraTransBlock, self).__init__()
        # Lightweight MHSA
        self.SlayerNorm = nn.LayerNorm(in_channels, eps=1e-6)
        self.ElayerNorm = nn.LayerNorm(in_channels, eps=1e-6)
        self.stage = stage
        self.lmhsa = Dual_axis(img_size, in_channels, d_h, d_v, d_w, num_heads, 0.0, self.stage)
        # Inverted Residual FFN
        self.irffn = FFN_MultiLN(in_channels, img_size, R)
        self.img_size = img_size

    def forward(self, x):
        q = x[0]
        k = x[1]
        v = x[2]
        x_pre = v  # (B, N, H)
        b, c, h, w = v.shape
        q = q.view(b, c, h * w).permute(0, 2, 1).contiguous()
        k = k.view(b, c, h * w).permute(0, 2, 1).contiguous()
        v = v.view(b, c, h * w).permute(0, 2, 1).contiguous()

        q = self.SlayerNorm(q)
        k = self.SlayerNorm(k)
        v = self.SlayerNorm(v)

        q = q.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        k = k.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        v = v.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        x = self.lmhsa(q, k, v)
        x = x_pre + x

        x_pre = x
        x = x.view(b, c, h * w).permute(0, 2, 1).contiguous()
        x = self.ElayerNorm(x)
        x = self.irffn(x)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        x = x_pre + x
        # print(self.stage)
        xx = (x, x, x)
        return xx


class ParallEncoder(nn.Module):
    def __init__(self):
        super(ParallEncoder, self).__init__()
        self.Encoder1 = UEncoder()
        self.Encoder2 = TransEncoder()
        self.fusion_module = nn.ModuleList()
        self.num_module = 4
        self.channel_list = [64, 128, 256, 512]
        self.fusion_list = [128, 256, 512, 512]

    def forward(self, x, boundary):
        skips = []
        features = self.Encoder1(x)
        feature_trans = self.Encoder2(x, features)

        skips.extend(features[:2])
        for i in range(self.num_module):
            skips.append(feature_trans[i])
        # skips = features
        return skips, features[1]


class TransEncoder(nn.Module):
    def __init__(self):
        super(TransEncoder, self).__init__()
        self.block_layer = [2, 2, 2, 2]
        self.size = [56, 28, 14, 7]
        self.channels = [128, 256, 512, 512]
        self.R = 2
        stage1 = []
        for _ in range(self.block_layer[0]):
            stage1.append(
                IntraTransBlock(
                    img_size=self.size[0],
                    in_channels=self.channels[0],
                    stride=2,
                    d_h=self.channels[0] // 8,
                    d_v=self.channels[0] // 8,
                    d_w=self.channels[0] // 8,
                    num_heads=8,
                    R=self.R,
                    stage=_
                )
            )
        self.stage1 = nn.Sequential(*stage1)
        stage2 = []
        for _ in range(self.block_layer[1]):
            stage2.append(
                IntraTransBlock(
                    img_size=self.size[1],
                    in_channels=self.channels[1],
                    stride=2,
                    d_h=self.channels[1] // 8,
                    d_v=self.channels[1] // 8,
                    d_w=self.channels[1] // 8,
                    num_heads=8,
                    R=self.R,
                    stage=_
                )
            )
        self.stage2 = nn.Sequential(*stage2)
        stage3 = []
        for _ in range(self.block_layer[2]):
            stage3.append(
                IntraTransBlock(
                    img_size=self.size[2],
                    in_channels=self.channels[2],
                    stride=2,
                    d_h=self.channels[2] // 8,
                    d_v=self.channels[2] // 8,
                    d_w=self.channels[2] // 8,
                    num_heads=8,
                    R=self.R,
                    stage=_
                )
            )
        self.stage3 = nn.Sequential(*stage3)
        stage4 = []
        for _ in range(self.block_layer[3]):
            stage4.append(
                IntraTransBlock(
                    img_size=self.size[3],
                    in_channels=self.channels[3],
                    stride=1,
                    d_h=self.channels[3] // 8,
                    d_v=self.channels[3] // 8,
                    d_w=self.channels[3] // 8,
                    num_heads=8,
                    R=self.R,
                    stage=_
                )
            )
        self.down_end = ConvBNReLU(self.channels[-1], self.channels[-1], 2, 2, padding=0)
        self.stage4 = nn.Sequential(*stage4)
        self.downlayers = nn.ModuleList()
        for i in range(len(self.block_layer) - 2):
            self.downlayers.append(
                ConvBNReLU(self.channels[i], self.channels[i] * 2, 2, 2, padding=0)
            )
        self.squeelayers = nn.ModuleList()

        for i in range(len(self.block_layer) - 1):
            self.squeelayers.append(
                nn.Conv2d(self.channels[i] * 2, self.channels[i] * 1, 1, 1)
            )
        self.squeeze_final = nn.Conv2d(self.channels[-1] * 2, self.channels[-1], 1, 1)

        self.stem = nn.Conv2d(3, self.channels[0], kernel_size=4, stride=4)
        self.ln = nn.LayerNorm(self.channels[0], eps=1e-6)

    def forward(self, x, x_cnn):
        _, _, feature0, feature1, feature2, feature3 = x_cnn
        q0 = feature0
        # k0 = feature0
        # v0 = feature0
        k0 = self.stem(x)
        k0 = k0.permute(0, 2, 3, 1)
        k0 = self.ln(k0)
        k0 = k0.permute(0, 3, 1, 2)
        v0 = torch.cat((q0, k0), dim=1)
        v0 = self.squeelayers[0](v0)
        x0 = (q0, k0, v0)
        feature0_trans = v0 + self.stage1(x0)[0]  # (56, 56, 256)
        feature0_trans_down = self.downlayers[0](feature0_trans)  # (28, 28, 512)

        q1 = feature1
        k1 = feature0_trans_down
        v1 = torch.cat((q1, feature0_trans_down), dim=1)
        v1 = self.squeelayers[1](v1)
        x1 = (q1, k1, v1)
        feature1_trans = v1 + self.stage2(x1)[0]

        feature1_trans_down = self.downlayers[1](feature1_trans)

        q2 = feature2
        k2 = feature1_trans_down
        v2 = torch.cat((q2, feature1_trans_down), dim=1)
        v2 = self.squeelayers[2](v2)
        x2 = (q2, k2, v2)
        feature2_trans = v2 + self.stage3(x2)[0]

        feature2_trans_down = self.down_end(feature2_trans)

        q3 = feature3
        k3 = feature2_trans_down
        v3 = torch.cat((q3, feature2_trans_down), dim=1)
        v3 = self.squeeze_final(v3)
        x3 = (q3, k3, v3)
        feature3_trans = v3 + self.stage4(x3)[0]

        return [feature0_trans, feature1_trans, feature2_trans, feature3_trans]


def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)
    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input


class Conv2dRe(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
                 groups=1, bias=True):
        super(Conv2dRe, self).__init__()
        self.bcr = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, groups=groups, bias=bias),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.bcr(x)


class IBD(nn.Module):
    def __init__(self, n_classes):
        super(IBD, self).__init__()
        #         self.reduce1 = ConvBNReLU(512, 128, kernel_size=1, padding=0)
        #         self.reduce2 = ConvBNReLU(256, 64, kernel_size=1, padding=0)
        #         self.reduce3 = ConvBNReLU(128, 32, kernel_size=1, padding=0)
        #         self.reduce4 = ConvBNReLU(64, 16, kernel_size=1, padding=0)
        #         self.reduce5 = ConvBNReLU(32, 8, kernel_size=1, padding=0)
        self.conv1 = nn.Sequential(
            Conv2dRe(256 + 256, 256),
            Conv2dRe(256, 256)
        )
        self.conv2 = nn.Sequential(
            Conv2dRe(128 + 128, 128),
            Conv2dRe(128, 128)
        )
        self.conv3 = nn.Sequential(
            Conv2dRe(64 + 64, 64),
            Conv2dRe(64, 64)
        )
        self.conv4 = nn.Sequential(
            Conv2dRe(32 + 32, 32),
            Conv2dRe(32, 32)
        )

        self.cr1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )
        self.cr2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.cr3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.cr4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )

        self.out1 = nn.Conv2d(32, 1, 3, 1, 1)
        self.out2 = nn.Conv2d(32, n_classes, 3, 1, 1)

        self.conv_out_1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.conv_out_2 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.conv_out_3 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.conv_out_4 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.conv_out_5 = nn.Conv2d(3, 1, kernel_size=3, padding=1)

        self.pool1 = nn.AvgPool2d(2)
        self.pool2 = nn.AvgPool2d(2)
        self.pool3 = nn.AvgPool2d(2)
        self.pool4 = nn.AvgPool2d(2)

        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x1, x2, x3, x4, x5):
        x1x = self.up1(x1)
        x1x = self.cr1(x1x)
        out = torch.cat((x1x, x2), dim=1)
        out1 = self.conv1(out)

        out2 = self.up2(out1)
        out2 = self.cr2(out2)
        out2 = torch.cat((out2, x3), dim=1)
        out2 = self.conv2(out2)

        out3 = self.up3(out2)
        out3 = self.cr3(out3)
        out3 = torch.cat((out3, x4), dim=1)
        out3 = self.conv3(out3)

        out4 = self.up4(out3)
        out4 = self.cr4(out4)
        out4 = torch.cat((out4, x5), dim=1)
        out4 = self.conv4(out4)

        out_o1 = self.out1(out4)
        out_o2 = self.out2(out4)

        avg_out1 = torch.mean(x1, dim=1, keepdim=True)
        s1 = self.conv_out_1(avg_out1)

        avg_out2 = torch.mean(x2, dim=1, keepdim=True)
        avg_out21 = torch.mean(out1, dim=1, keepdim=True)
        s2 = torch.cat([avg_out2, avg_out21], dim=1)
        s2 = self.conv_out_2(s2)

        avg_out3 = torch.mean(x3, dim=1, keepdim=True)
        avg_out31 = torch.mean(out2, dim=1, keepdim=True)
        s3 = torch.cat([avg_out3, avg_out31], dim=1)
        s3 = self.conv_out_3(s3)

        avg_out4 = torch.mean(x4, dim=1, keepdim=True)
        avg_out41 = torch.mean(out3, dim=1, keepdim=True)
        s4 = torch.cat([avg_out4, avg_out41], dim=1)
        s4 = self.conv_out_4(s4)

        avg_out5 = torch.mean(x5, dim=1, keepdim=True)
        avg_out51 = torch.mean(out4, dim=1, keepdim=True)
        s5 = torch.cat([avg_out5, avg_out51], dim=1)
        s5 = torch.cat([s5, out_o1], dim=1)
        s5 = self.conv_out_5(s5)

        s = [s1, s2, s3, s4, s5]
        return out_o1, out_o2, s


# class IBD(nn.Module):
#     def __init__(self, n_classes):
#         super(IBD, self).__init__()
#         self.reduce1 = ConvBNReLU(512, 128, kernel_size=1, padding=0)
#         self.reduce2 = ConvBNReLU(256, 64, kernel_size=1, padding=0)
#         self.reduce3 = ConvBNReLU(128, 32, kernel_size=1, padding=0)
#         self.reduce4 = ConvBNReLU(64, 16, kernel_size=1, padding=0)
#         self.reduce5 = ConvBNReLU(32, 8, kernel_size=1, padding=0)
#         self.conv = ConvBNReLU(c_in=128 + 64 + 32 + 16 + 8, c_out=32, kernel_size=3)
#         self.out1 = nn.Conv2d(32, 1, 1)
#         self.out2 = nn.Conv2d(32, n_classes, 1)
#         self.conv_out_1 = nn.Conv2d(2, 1, 1)
#         self.conv_out_2 = nn.Conv2d(2, 1, 1)
#         self.conv_out_3 = nn.Conv2d(2, 1, 1)
#         self.conv_out_4 = nn.Conv2d(2, 1, 1)
#         self.conv_out_5 = nn.Conv2d(2, 1, 1)

#         self.pool1 = nn.MaxPool2d(2)
#         self.pool2 = nn.MaxPool2d(4)
#         self.pool3 = nn.MaxPool2d(8)
#         self.pool4 = nn.MaxPool2d(16)

#         self.up1 = nn.UpsamplingBilinear2d(scale_factor=16)
#         self.up2 = nn.UpsamplingBilinear2d(scale_factor=8)
#         self.up3 = nn.UpsamplingBilinear2d(scale_factor=4)
#         self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)

#     def forward(self, x1, x2, x3, x4, x5):
#         size = x5.size()[2:]
#         x1 = self.reduce1(x1)
#         x2 = self.reduce2(x2)
#         x3 = self.reduce3(x3)
#         x4 = self.reduce4(x4)
#         x5 = self.reduce5(x5)
# #         x1x = F.interpolate(x1, size, mode='bilinear', align_corners=False)
# #         x2x = F.interpolate(x2, size, mode='bilinear', align_corners=False)
# #         x3x = F.interpolate(x3, size, mode='bilinear', align_corners=False)
# #         x4x = F.interpolate(x4, size, mode='bilinear', align_corners=False)
#         x1x = self.up1(x1)
#         x2x = self.up2(x2)
#         x3x = self.up3(x3)
#         x4x = self.up4(x4)
#         out = torch.cat((x1x, x2x), dim=1)
#         out = torch.cat((out, x3x), dim=1)
#         out = torch.cat((out, x4x), dim=1)
#         out = torch.cat((out, x5), dim=1)
#         out = self.conv(out)
#         out1 = self.out1(out)
#         out2 = self.out2(out)

#         avg_out1 = torch.mean(x1, dim=1, keepdim=True)
#         s1 = torch.cat([avg_out1, self.pool4(out1)], dim=1)
#         s1 = self.conv_out_1(s1)

#         avg_out2 = torch.mean(x2, dim=1, keepdim=True)
#         s2 = torch.cat([avg_out2, self.pool3(out1)], dim=1)
#         s2 = self.conv_out_2(s2)

#         avg_out3 = torch.mean(x3, dim=1, keepdim=True)
#         s3 = torch.cat([avg_out3, self.pool2(out1)], dim=1)
#         s3 = self.conv_out_3(s3)

#         avg_out4 = torch.mean(x4, dim=1, keepdim=True)
#         s4 = torch.cat([avg_out4, self.pool1(out1)], dim=1)
#         s4 = self.conv_out_4(s4)

#         avg_out5 = torch.mean(x5, dim=1, keepdim=True)
#         s5 = torch.cat([avg_out5, out1], dim=1)
#         s5 = self.conv_out_5(s5)

#         s = [s1, s2, s3, s4, s5]
#         return out1, out2, s


# class BAG(nn.Module):
#     def __init__(self, in_channels):
#         super(BAG, self).__init__()
#         self.channels = in_channels
#         self.conv_skip = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels,
#                       kernel_size=1, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU()
#         )
#         self.conv_deep = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels,
#                       kernel_size=1, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU()
#         )
#         self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.cbr1 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU()
#         )
#         self.cbr2 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU()
#         )
#         self.conv_out = nn.Sequential(
#             nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU()
#         )

#     def forward(self, skip, deep, edge_att):
#         skip = self.cbr1(skip)
#         deep = self.cbr2(self.up1(deep))

#         skip_add = self.conv_skip((1 - edge_att) * deep + skip)
#         deep_add = self.conv_deep(deep + edge_att * skip)
#         out = self.conv_out(torch.cat((skip_add, deep_add), dim=1))
#         return out
# return torch.cat((skip_add, deep_add), dim=1)

# class IBD(nn.Module):
#     def __init__(self, n_classes):
#         super(IBD, self).__init__()
#         self.reduce1 = ConvBNReLU(512, 128, kernel_size=1, padding=0)
#         self.reduce3 = ConvBNReLU(128, 64, kernel_size=1, padding=0)
#         self.reduce5 = ConvBNReLU(32, 32, kernel_size=1, padding=0)
#         self.conv = ConvBNReLU(c_in=128 + 64 + 32, c_out=64, kernel_size=3)
#         self.out1 = nn.Conv2d(64, 1, 1)
#         self.out2 = nn.Conv2d(64, n_classes, 1)

#     def forward(self, x1, x3, x5):
#         size = x5.size()[2:]
#         x1 = self.reduce1(x1)
#         x3 = self.reduce3(x3)
#         x5 = self.reduce5(x5)
#         x1 = F.interpolate(x1, size, mode='bilinear', align_corners=False)
#         x3 = F.interpolate(x3, size, mode='bilinear', align_corners=False)
#         out = torch.cat((x1, x3), dim=1)
#         out = torch.cat((out, x5), dim=1)
#         out = self.conv(out)
#         out1 = self.out1(out)
#         out2 = self.out2(out)

#         return out1, out2


class BAG(nn.Module):
    def __init__(self, in_channels):
        super(BAG, self).__init__()
        self.channels = in_channels
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.conv_deep = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, skip, deep, edge_att):
        deep = self.up1(deep)

        skip_add = self.conv_skip((1 - edge_att) * deep + skip)
        deep_add = self.conv_deep(deep + edge_att * skip)

        return skip_add + deep_add
        # return torch.cat((skip_add, deep_add), dim=1)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DGCANet(nn.Module):
    def __init__(self, n_classes):
        super(DGCANet, self).__init__()
        self.p_encoder = ParallEncoder()
        self.out_size = (224, 224)
        self.encoder_channels = [512, 256, 128, 64, 32]

        self.sobel_x5, self.sobel_y5 = get_sobel(32, 1)
        self.sobel_x4, self.sobel_y4 = get_sobel(64, 1)
        self.sobel_x3, self.sobel_y3 = get_sobel(128, 1)
        self.sobel_x2, self.sobel_y2 = get_sobel(256, 1)
        self.sobel_x1, self.sobel_y1 = get_sobel(512, 1)

        self.ibd = IBD(n_classes)

        self.bag1 = BAG(in_channels=512)
        self.bag2 = BAG(in_channels=256)
        self.bag3 = BAG(in_channels=128)
        self.bag4 = BAG(in_channels=64)
        self.bag5 = BAG(in_channels=32)

        self.decoder1 = DecoderBlock(self.encoder_channels[0] + self.encoder_channels[0], self.encoder_channels[1])
        self.decoder2 = DecoderBlock(self.encoder_channels[1] + self.encoder_channels[1], self.encoder_channels[2])
        self.decoder3 = DecoderBlock(self.encoder_channels[2] + self.encoder_channels[2], self.encoder_channels[3])
        self.decoder4 = DecoderBlock(self.encoder_channels[3] + self.encoder_channels[3], self.encoder_channels[4])
        self.segmentation_head = SegmentationHead(
            in_channels=32,
            out_channels=n_classes,
            kernel_size=3,
        )
        self.segmentation_head1 = SegmentationHead(
            in_channels=32,
            out_channels=n_classes,
            kernel_size=3,
        )
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(4)
        self.pool3 = nn.MaxPool2d(8)
        self.pool4 = nn.MaxPool2d(16)
        self.decoder_final = DecoderBlock(in_channels=32 * 2, out_channels=32)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        encoder_skips, feature1 = self.p_encoder(x, x)

        e1 = run_sobel(self.sobel_x1, self.sobel_y1, encoder_skips[-2])  # 512  14
        e2 = run_sobel(self.sobel_x2, self.sobel_y2, encoder_skips[-3])  # 256  28
        e3 = run_sobel(self.sobel_x3, self.sobel_y3, encoder_skips[-4])  # 128  56
        e4 = run_sobel(self.sobel_x4, self.sobel_y4, encoder_skips[-5])  # 64  112
        e5 = run_sobel(self.sobel_x5, self.sobel_y5, encoder_skips[-6])  # 32  224
        edge, edge_output, s = self.ibd(e1, e2, e3, e4, e5)
        # edge, edge_output = self.ibd(e1, e3, e5)

        edge_att1 = torch.sigmoid(s[0])
        bag1 = self.bag1(encoder_skips[-2], encoder_skips[-1], edge_att1)
        x1_up = self.decoder1(encoder_skips[-1], bag1)

        edge_att2 = torch.sigmoid(s[1])
        bag2 = self.bag2(encoder_skips[-3], x1_up, edge_att2)
        x2_up = self.decoder2(x1_up, bag2)

        edge_att3 = torch.sigmoid(s[2])
        bag3 = self.bag3(encoder_skips[-4], x2_up, edge_att3)
        x3_up = self.decoder3(x2_up, bag3)

        edge_att4 = torch.sigmoid(s[3])
        bag4 = self.bag4(encoder_skips[-5], x3_up, edge_att4)
        x4_up = self.decoder4(x3_up, bag4)

        edge_att5 = torch.sigmoid(s[4])
        bag5 = self.bag5(encoder_skips[-6], x4_up, edge_att5)
        x_final = self.decoder_final(x4_up, bag5)

        logits = self.segmentation_head(x_final)
        edge_decoder = self.segmentation_head1(x_final)

        #         edge_att = torch.sigmoid(edge)  # 1 * 56 * 56
        #         edge_att1 = self.pool4(edge_att)
        #         sbg1 = self.bag1(encoder_skips[-2], encoder_skips[-1], edge_att1)
        #         x1_up = self.decoder1(encoder_skips[-1], sbg1)

        #         edge_att2 = self.pool3(edge_att)
        #         sbg2 = self.bag2(encoder_skips[-3], x1_up, edge_att2)
        #         x2_up = self.decoder2(x1_up, sbg2)

        #         edge_att3 = self.pool2(edge_att)
        #         sbg3 = self.bag3(encoder_skips[-4], x2_up, edge_att3)
        #         x3_up = self.decoder3(x2_up, sbg3)

        #         edge_att4 = self.pool1(edge_att)
        #         sbg4 = self.bag4(encoder_skips[-5], x3_up, edge_att4)
        #         x4_up = self.decoder4(x3_up, sbg4)
        #         edge_att5 = edge_att
        #         sbg5 = self.bag5(encoder_skips[-6], x4_up, edge_att5)
        #         x_final = self.decoder_final(x4_up, sbg5)

        #         logits = self.segmentation_head(x_final)
        #         edge_decoder = self.segmentation_head1(x_final)

        return logits, edge_decoder, edge_output


model = DGCANet(n_classes=9)
inout = torch.ones((1, 1, 224, 224))
# print(model)
flops, params = profile(model, (inout,))
print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))
