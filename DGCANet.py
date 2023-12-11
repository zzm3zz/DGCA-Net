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
        self.res1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.res2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.res3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.res4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.res5 = DoubleConv(512, 1024)
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

        if self.stage == 0:
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
            # result = result.view(b, h * w, self.heads * self.d_v).contiguous()
            
        elif self.stage == 1:
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
            result_h = result_h.view(v_b, self.heads, v_h*v_w, self.d_v).contiguous()  # b, heads, h*w, c
            result_w = result_w.permute(0, 1, 3, 2)  # b, heads, c*h, w
            result_w = result_w.view(v_b, self.heads, self.d_v, v_h, v_w).contiguous()  # b, heads, c, h, w
            result_w = result_w.view(v_b, self.heads, self.d_v, v_h*v_w).contiguous()  # b, heads, c, h*w
            result_w = result_w.permute(0, 1, 3, 2).contiguous()  # b, heads, w*h, c
            result = result_h + result_w  # b, heads, w*h, c
            result = result.permute(0, 2, 1, 3).contiguous()  # b, n, heads, c
            # print(result.shape)
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

        if self.stage != 1 and self.img_size != 7:
            xx = (x, x, x)
            return xx
        else:
            return x


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


class ParallEncoder(nn.Module):
    def __init__(self):
        super(ParallEncoder, self).__init__()
        self.Encoder1 = UEncoder()
        self.Encoder2 = TransEncoder()
        self.fusion_module = nn.ModuleList()
        self.num_module = 4
        self.channel_list = [128, 256, 512, 1024]
        self.fusion_list = [256, 512, 1024, 1024]

        self.squeelayers = nn.ModuleList()
        for i in range(self.num_module):
            self.squeelayers.append(
                nn.Conv2d(self.fusion_list[i] * 2, self.fusion_list[i], 1, 1)
            )

    def forward(self, x, boundary):
        skips = []
        features = self.Encoder1(x)
        feature_trans = self.Encoder2(features)

        skips.extend(features[:2])
        for i in range(self.num_module):
            skip = self.squeelayers[i](torch.cat((feature_trans[i], features[i + 2]), dim=1))
            skips.append(skip)
        return skips, features[1]

    
class conv(nn.Module):
    def __init__(self, channels_in, channels, nums):
        super(conv, self).__init__()
        self.channels_in = channels_in
        self.channels = channels
        self.nums = nums

        self.linear_1 = nn.Linear(self.channels_in, self.channels)
        self.conv2d_1 = nn.Conv2d(self.channels, self.channels, 1)
        self.relu = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(self.channels, self.channels, 1)
        self.linear_2 = nn.Linear(self.channels, self.channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):  # b c h w
        if self.nums == 2:
            x = torch.cat((features[0], features[1]), dim=1)
        if self.nums == 3:
            x = torch.cat([features[0], features[1]], dim=1)
            x = torch.cat([x, features[2]], dim=1)
        x = x.permute(0, 2, 3, 1)  # b h w c
        # print(x.shape)
        x = self.linear_1(x)
        x = x.permute(0, 3, 1, 2)  # b c h w
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = x.permute(0, 2, 3, 1)  # b h w c
        x = self.linear_2(x)
        x = x.permute(0, 3, 1, 2)  # b c h w

        return x
    
    
class TransEncoder(nn.Module):
    def __init__(self):
        super(TransEncoder, self).__init__()
        self.block_layer = [2, 2, 2, 1]
        self.size = [56, 28, 14, 7]
        self.channels = [256, 512, 1024, 1024]
        self.R = 4
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
                    stage=_
                )
            )
        self.down_end = ConvBNReLU(self.channels[-1], self.channels[-1], 2, 2, padding=0)
        self.stage4 = nn.Sequential(*stage4)
        self.downlayers = nn.ModuleList()
        for i in range(len(self.block_layer) - 1):
            self.downlayers.append(
                ConvBNReLU(self.channels[i], self.channels[i] * 2, 2, 2, padding=0)
            )
        self.pool = nn.MaxPool2d(2)
        self.conv1x1 = Conv2dReLU(
            128,
            256,
            kernel_size=1,
            padding=0,
        )
        self.squeelayers = nn.ModuleList()

        for i in range(len(self.block_layer) - 2):
            self.squeelayers.append(
                nn.Conv2d(self.channels[i] * 4, self.channels[i] * 2, 1, 1)
            )
        
        self.squeeze_final = nn.Conv2d(self.channels[-1] * 2, self.channels[-1], 1, 1)

    def forward(self, x):
        _, feature00, feature0, feature1, feature2, feature3 = x
        feature00 = self.pool(feature00)
        feature00 = self.conv1x1(feature00)
        q0 = feature0
        k0 = feature00
        v0 = feature0
        # v0 = self.fusion0([feature0, feature_edge_0])
        x0 = (q0, k0, v0)
        feature0_trans = v0 + self.stage1(x0)  # (56, 56, 256)
        feature0_trans_down = self.downlayers[0](feature0_trans)  # (28, 28, 512)

        q1 = feature1
        # q1 = self.fusion1([feature1, feature_edge_1])
        k1 = feature0_trans_down
        v1 = torch.cat((q1, feature0_trans_down), dim=1)
        v1 = self.squeelayers[0](v1)
        x1 = (q1, k1, v1)
        feature1_trans = v1 + self.stage2(x1)

        feature1_trans_down = self.downlayers[1](feature1_trans)

        q2 = feature2
        # q2 = self.fusion2([feature2, feature_edge_2])
        k2 = feature1_trans_down
        v2 = torch.cat((q2, feature1_trans_down), dim=1)
        v2 = self.squeelayers[1](v2)
        x2 = (q2, k2, v2)
        feature2_trans = v2 + self.stage3(x2)

        feature2_trans_down = self.down_end(feature2_trans)

        q3 = feature3
        # q3 = self.fusion3([feature3, feature_edge_3])
        k3 = feature2_trans_down
        v3 = torch.cat((q3, feature2_trans_down), dim=1)
        v3 = self.squeeze_final(v3)
        x3 = (q3, k3, v3)
        feature3_trans = v3 + self.stage4(x3)

        return [feature0_trans, feature1_trans, feature2_trans, feature3_trans]
    
def get_sobel(in_chan, out_chan):
    '''
    filter_x = np.array([
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3],
    ]).astype(np.float32)
    filter_y = np.array([
        [3, 10, 3],
        [0, 0, 0],
        [-3, -10, -3],
    ]).astype(np.float32)
    '''
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


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

    
class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    

class BED(nn.Module):
    def __init__(self, n_classes):
        super(BED, self).__init__()
        self.reduce1 = Conv1x1(1024, 256)
        self.reduce3 = Conv1x1(256, 256)
        self.reduce5 = Conv1x1(64, 64)
        self.block = nn.Sequential(
            ConvBNR(320 + 256, 256, 3),
            ConvBNR(256, 256, 3))
        self.out1 = nn.Conv2d(256, 1, 1)
        self.out2 = nn.Conv2d(256, n_classes, 1)

    def forward(self, x1, x3, x5):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        # x1 = F.interpolate(x1, size, mode='bilinear', align_corners=False)
        x3 = self.reduce3(x3)
        x3 = F.interpolate(x3, size, mode='bilinear', align_corners=False)
        x5 = self.reduce5(x5)
        x5 = F.interpolate(x5, size, mode='bilinear', align_corners=False) 
        out = torch.cat((x1, x3), dim=1)
        out = torch.cat((out, x5), dim=1)
        out = self.block(out)
        out1 = self.out1(out)
        out2 = self.out2(out)
        
        return out1, out2
    
    
class SBG(nn.Module):
    def __init__(self, in_channels):
        super(SBG, self).__init__()
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
    
    
class DGCANet(nn.Module):
    def __init__(self, n_classes):
        super(DGCANet, self).__init__()
        self.p_encoder = ParallEncoder()
        self.out_size = (224, 224)
        self.encoder_channels = [1024, 512, 256, 128, 64]
        
        self.sobel_x5, self.sobel_y5 = get_sobel(64, 1)
        self.sobel_x3, self.sobel_y3 = get_sobel(256, 1)
        self.sobel_x1, self.sobel_y1 = get_sobel(1024, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(4)
        
        self.bed = BED(n_classes)
        
        self.sbg1 = SBG(in_channels=1024)
        self.sbg2 = SBG(in_channels=512)
        self.sbg3 = SBG(in_channels=256)
        self.sbg4 = SBG(in_channels=128)
        self.sbg5 = SBG(in_channels=64)
        
        self.decoder1 = DecoderBlock(self.encoder_channels[0] + self.encoder_channels[0], self.encoder_channels[1])
        self.decoder2 = DecoderBlock(self.encoder_channels[1] + self.encoder_channels[1], self.encoder_channels[2])
        self.decoder3 = DecoderBlock(self.encoder_channels[2] + self.encoder_channels[2], self.encoder_channels[3])
        self.decoder4 = DecoderBlock(self.encoder_channels[3] + self.encoder_channels[3], self.encoder_channels[4])
        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=n_classes,
            kernel_size=3,
        )
        self.segmentation_head1 = SegmentationHead(
            in_channels=64,
            out_channels=n_classes,
            kernel_size=3,
        )
        
        self.decoder_final = DecoderBlock(in_channels=64 * 2, out_channels=64)        

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        encoder_skips, feature1 = self.p_encoder(x, x)
        
        e1 = run_sobel(self.sobel_x1, self.sobel_y1, encoder_skips[-2])  # 1024  14
        e3 = run_sobel(self.sobel_x3, self.sobel_y3, encoder_skips[-4])  # 256  56
        e5 = run_sobel(self.sobel_x5, self.sobel_y5, encoder_skips[-6])  # 64  224
        edge, edge_output = self.bed(e1, e3, e5)

        edge_att = torch.sigmoid(edge)  # 1 * 56 * 56
        
        # edge_att1 = F.interpolate(edge_att, encoder_skips[-2].size()[2:], mode='bilinear', align_corners=False)
        edge_att1 = edge_att
        sbg1 = self.sbg1(encoder_skips[-2], encoder_skips[-1], edge_att1)
        x1_up = self.decoder1(encoder_skips[-1], sbg1)
        
        edge_att2 = F.interpolate(edge_att, encoder_skips[-3].size()[2:], mode='bilinear', align_corners=False)
        sbg2 = self.sbg2(encoder_skips[-3], x1_up, edge_att2)
        x2_up = self.decoder2(x1_up, sbg2)
        
        edge_att3 = F.interpolate(edge_att, encoder_skips[-4].size()[2:], mode='bilinear', align_corners=False)
        sbg3 = self.sbg3(encoder_skips[-4], x2_up, edge_att3)
        x3_up = self.decoder3(x2_up, sbg3)
        
        edge_att4 = F.interpolate(edge_att, encoder_skips[-5].size()[2:], mode='bilinear', align_corners=False)
        sbg4 = self.sbg4(encoder_skips[-5], x3_up, edge_att4)
        x4_up = self.decoder4(x3_up, sbg4)
        
        edge_att5 = F.interpolate(edge_att, encoder_skips[-6].size()[2:], mode='bilinear', align_corners=False)
        sbg5 = self.sbg5(encoder_skips[-6], x4_up, edge_att5)
        x_final = self.decoder_final(x4_up, sbg5)

        logits = self.segmentation_head(x_final)
        edge_decoder = self.segmentation_head1(x_final)
        
        edge_output = F.interpolate(edge_output, encoder_skips[-6].size()[2:], mode='bilinear', align_corners=False)
        # print(edge_output.shape)
        return logits, edge_decoder


model = DGCANet(n_classes=9)
inout = torch.ones((1, 1, 224, 224))

flops, params = profile(model, (inout,))
print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))
