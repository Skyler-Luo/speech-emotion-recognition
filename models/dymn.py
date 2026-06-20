"""
Dynamic MobileNet (DyMN) for Audio Classification
Reference: https://github.com/fschmid56/EfficientAT
"""
import math
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple
import urllib.parse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.hub import load_state_dict_from_url
from torchvision.ops.misc import Conv2dNormActivation

from models.mn import InvertedResidual, InvertedResidualConfig


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def cnn_out_size(in_size, padding, dilation, kernel, stride):
    s = in_size + 2 * padding - dilation * (kernel - 1) - 1
    return math.floor(s / stride + 1)


class DynamicInvertedResidualConfig:
    def __init__(self, input_channels, kernel, expanded_channels, out_channels,
                 use_dy_block, activation, stride, dilation, width_mult):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_dy_block = use_dy_block
        self.use_hs = activation == "HS"
        self.use_se = False
        self.stride = stride
        self.dilation = dilation
        self.width_mult = width_mult

    @staticmethod
    def adjust_channels(channels, width_mult):
        return make_divisible(channels * width_mult, 8)

    def out_size(self, in_size):
        padding = (self.kernel - 1) // 2 * self.dilation
        return cnn_out_size(in_size, padding, self.dilation, self.kernel, self.stride)


class DynamicConv(nn.Module):
    def __init__(self, in_channels, out_channels, context_dim, kernel_size,
                 stride=1, dilation=1, padding=0, groups=1, att_groups=1,
                 bias=False, k=4, temp_schedule=(30, 1, 1, 0.05)):
        super().__init__()
        assert in_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.k = k
        self.T_max, self.T_min, self.T0_slope, self.T1_slope = temp_schedule
        self.temperature = self.T_max
        self.att_groups = att_groups
        self.residuals = nn.Sequential(nn.Linear(context_dim, k * self.att_groups))

        weight = torch.randn(k, out_channels, in_channels // groups, kernel_size, kernel_size)
        if bias:
            self.bias = nn.Parameter(torch.zeros(k, out_channels), requires_grad=True)
        else:
            self.bias = None
        self._initialize_weights(weight, self.bias)
        weight = weight.view(1, k, att_groups, out_channels, in_channels // groups, kernel_size, kernel_size)
        weight = weight.transpose(1, 2).view(1, self.att_groups, self.k, -1)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def _initialize_weights(self, weight, bias):
        init_func = partial(nn.init.kaiming_normal_, mode="fan_out")
        for i in range(self.k):
            init_func(weight[i])
            if bias is not None:
                nn.init.zeros_(bias[i])

    def forward(self, x, g=None):
        b, c, f, t = x.size()
        g_c = g[0].view(b, -1)
        residuals = self.residuals(g_c).view(b, self.att_groups, 1, -1)
        attention = F.softmax(residuals / self.temperature, dim=-1)
        aggregate_weight = (attention @ self.weight).transpose(1, 2).reshape(
            b, self.out_channels, self.in_channels // self.groups,
            self.kernel_size, self.kernel_size
        )
        aggregate_weight = aggregate_weight.view(
            b * self.out_channels, self.in_channels // self.groups,
            self.kernel_size, self.kernel_size
        )
        x = x.view(1, -1, f, t)
        if self.bias is not None:
            aggregate_bias = torch.mm(attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias,
                              stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * b)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None,
                              stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * b)
        return output.view(b, self.out_channels, output.size(-2), output.size(-1))

    def update_params(self, epoch):
        t0 = self.T_max - self.T0_slope * epoch
        t1 = 1 + self.T1_slope * (self.T_max - 1) / self.T0_slope - self.T1_slope * epoch
        self.temperature = max(t0, t1, self.T_min)
        print(f"Setting temperature for attention over kernels to {self.temperature}")


class DyReLU(nn.Module):
    def __init__(self, channels, context_dim, M=2):
        super().__init__()
        self.channels = channels
        self.M = M
        self.coef_net = nn.Sequential(nn.Linear(context_dim, 2 * M))
        self.sigmoid = nn.Sigmoid()
        self.register_buffer('lambdas', torch.Tensor([1.] * M + [0.5] * M).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.] * (2 * M - 1)).float())

    def get_relu_coefs(self, x):
        theta = self.coef_net(x)
        return 2 * self.sigmoid(theta) - 1

    def forward(self, x, g):
        raise NotImplementedError


class DyReLUB(DyReLU):
    def __init__(self, channels, context_dim, M=2):
        super().__init__(channels, context_dim, M)
        self.coef_net[-1] = nn.Linear(context_dim, 2 * M * self.channels)

    def forward(self, x, g):
        assert x.shape[1] == self.channels and g is not None
        b, c, f, t = x.size()
        h_c = g[0].view(b, -1)
        theta = self.get_relu_coefs(h_c)
        relu_coefs = theta.view(-1, self.channels, 1, 1, 2 * self.M) * self.lambdas + self.init_v
        x_mapped = x.unsqueeze(-1) * relu_coefs[:, :, :, :, :self.M] + relu_coefs[:, :, :, :, self.M:]
        if self.M == 2:
            return torch.maximum(x_mapped[:, :, :, :, 0], x_mapped[:, :, :, :, 1])
        return torch.max(x_mapped, dim=-1)[0]


class CoordAtt(nn.Module):
    def forward(self, x, g):
        g_cf, g_ct = g[1], g[2]
        return x * g_cf.sigmoid() * g_ct.sigmoid()


class DynamicWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, g):
        return self.module(x)


class ContextGen(nn.Module):
    def __init__(self, context_dim, in_ch, exp_ch, norm_layer, stride: int = 1):
        super().__init__()
        self.joint_conv = nn.Conv2d(in_ch, context_dim, kernel_size=(1, 1), padding=0, bias=False)
        self.joint_norm = norm_layer(context_dim)
        self.joint_act = nn.Hardswish(inplace=True)
        self.conv_f = nn.Conv2d(context_dim, exp_ch, kernel_size=(1, 1), padding=0)
        self.conv_t = nn.Conv2d(context_dim, exp_ch, kernel_size=(1, 1), padding=0)
        if stride > 1:
            self.pool_f = nn.AvgPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0))
            self.pool_t = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, stride), padding=(0, 1))
        else:
            self.pool_f = nn.Sequential()
            self.pool_t = nn.Sequential()

    def forward(self, x, g):
        cf = F.adaptive_avg_pool2d(x, (None, 1))
        ct = F.adaptive_avg_pool2d(x, (1, None)).permute(0, 1, 3, 2)
        f, t = cf.size(2), ct.size(2)
        g_cat = self.joint_act(self.joint_norm(self.joint_conv(torch.cat([cf, ct], dim=2))))
        h_cf, h_ct = torch.split(g_cat, [f, t], dim=2)
        h_ct = h_ct.permute(0, 1, 3, 2)
        h_c = torch.mean(g_cat, dim=2, keepdim=True)
        return (h_c, self.conv_f(self.pool_f(h_cf)), self.conv_t(self.pool_t(h_ct)))


class DY_Block(nn.Module):
    def __init__(self, cnf: DynamicInvertedResidualConfig, context_ratio=4,
                 max_context_size=128, min_context_size=32, temp_schedule=(30, 1, 1, 0.05),
                 dyrelu_k=2, dyconv_k=4, no_dyrelu=False, no_dyconv=False,
                 no_ca=False, **kwargs):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        self.context_dim = np.clip(
            make_divisible(cnf.expanded_channels // context_ratio, 8),
            make_divisible(min_context_size * cnf.width_mult, 8),
            make_divisible(max_context_size * cnf.width_mult, 8)
        )
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        def make_conv1x1(in_c, out_c):
            if no_dyconv:
                return DynamicWrapper(nn.Conv2d(in_c, out_c, 1, bias=False))
            return DynamicConv(in_c, out_c, self.context_dim, 1, k=dyconv_k,
                               temp_schedule=temp_schedule, bias=False)

        if cnf.expanded_channels != cnf.input_channels:
            self.exp_conv = make_conv1x1(cnf.input_channels, cnf.expanded_channels)
            self.exp_norm = norm_layer(cnf.expanded_channels)
            self.exp_act = DynamicWrapper(activation_layer(inplace=True))
        else:
            self.exp_conv = DynamicWrapper(nn.Identity())
            self.exp_norm = nn.Identity()
            self.exp_act = DynamicWrapper(nn.Identity())

        stride = 1 if cnf.dilation > 1 else cnf.stride
        padding = (cnf.kernel - 1) // 2 * cnf.dilation
        if no_dyconv:
            self.depth_conv = DynamicWrapper(nn.Conv2d(
                cnf.expanded_channels, cnf.expanded_channels, cnf.kernel,
                groups=cnf.expanded_channels, stride=stride,
                dilation=cnf.dilation, padding=padding, bias=False
            ))
        else:
            self.depth_conv = DynamicConv(
                cnf.expanded_channels, cnf.expanded_channels, self.context_dim,
                cnf.kernel, k=dyconv_k, temp_schedule=temp_schedule,
                groups=cnf.expanded_channels, stride=stride,
                dilation=cnf.dilation, padding=padding, bias=False
            )
        self.depth_norm = norm_layer(cnf.expanded_channels)
        self.depth_act = (DynamicWrapper(activation_layer(inplace=True)) if no_dyrelu
                          else DyReLUB(cnf.expanded_channels, self.context_dim, M=dyrelu_k))
        self.ca = DynamicWrapper(nn.Identity()) if no_ca else CoordAtt()
        self.proj_conv = make_conv1x1(cnf.expanded_channels, cnf.out_channels)
        self.proj_norm = norm_layer(cnf.out_channels)
        self.context_gen = ContextGen(self.context_dim, cnf.input_channels,
                                      cnf.expanded_channels, norm_layer, stride)

    def forward(self, x, g=None):
        inp = x
        g = self.context_gen(x, g)
        x = self.exp_act(self.exp_norm(self.exp_conv(x, g)), g)
        x = self.ca(self.depth_act(self.depth_norm(self.depth_conv(x, g)), g), g)
        x = self.proj_norm(self.proj_conv(x, g))
        if self.use_res_connect:
            x += inp
        return x


model_url = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/"
model_dir = "weights"

pretrained_models = {
    "dymn04_im": urllib.parse.urljoin(model_url, "dymn04_im.pt"),
    "dymn10_im": urllib.parse.urljoin(model_url, "dymn10_im.pt"),
    "dymn20_im": urllib.parse.urljoin(model_url, "dymn20_im.pt"),
    "dymn04_as": urllib.parse.urljoin(model_url, "dymn04_as.pt"),
    "dymn10_as": urllib.parse.urljoin(model_url, "dymn10_as.pt"),
    "dymn20_as": urllib.parse.urljoin(model_url, "dymn20_as_mAP_493.pt"),
    "dymn20_as(1)": urllib.parse.urljoin(model_url, "dymn20_as.pt"),
    "dymn20_as(2)": urllib.parse.urljoin(model_url, "dymn20_as_mAP_489.pt"),
    "dymn20_as(3)": urllib.parse.urljoin(model_url, "dymn20_as_mAP_490.pt"),
    "dymn04_replace_se_as": urllib.parse.urljoin(model_url, "dymn04_replace_se_as.pt"),
    "dymn10_replace_se_as": urllib.parse.urljoin(model_url, "dymn10_replace_se_as.pt"),
}


class DyMN(nn.Module):
    def __init__(self, inverted_residual_setting: List[DynamicInvertedResidualConfig],
                 last_channel: int, num_classes: int = 527, head_type: str = "mlp",
                 block=None, norm_layer=None, dropout: float = 0.2,
                 in_conv_kernel: int = 3, in_conv_stride: int = 2, in_channels: int = 1,
                 context_ratio: int = 4, max_context_size: int = 128,
                 min_context_size: int = 32, dyrelu_k=2, dyconv_k=4,
                 no_dyrelu=False, no_dyconv=False, no_ca=False,
                 temp_schedule=(30, 1, 1, 0.05), **kwargs) -> None:
        super().__init__()
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, Sequence) and
                  all(isinstance(s, DynamicInvertedResidualConfig) for s in inverted_residual_setting)):
            raise TypeError("The inverted_residual_setting should be List[DynamicInvertedResidualConfig]")

        if block is None:
            block = DY_Block
        norm_layer = norm_layer if norm_layer is not None else partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        self.layers = nn.ModuleList()
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.in_c = Conv2dNormActivation(
            in_channels, firstconv_output_channels, kernel_size=in_conv_kernel,
            stride=in_conv_stride, norm_layer=norm_layer, activation_layer=nn.Hardswish,
        )

        for cnf in inverted_residual_setting:
            if cnf.use_dy_block:
                b = block(cnf, context_ratio=context_ratio,
                          max_context_size=max_context_size, min_context_size=min_context_size,
                          dyrelu_k=dyrelu_k, dyconv_k=dyconv_k, no_dyrelu=no_dyrelu,
                          no_dyconv=no_dyconv, no_ca=no_ca, temp_schedule=temp_schedule)
            else:
                b = InvertedResidual(cnf, None, norm_layer,
                                     partial(nn.BatchNorm2d, eps=0.001, momentum=0.01))
            self.layers.append(b)

        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        self.out_c = Conv2dNormActivation(
            lastconv_input_channels, lastconv_output_channels, kernel_size=1,
            norm_layer=norm_layer, activation_layer=nn.Hardswish,
        )

        self.head_type = head_type
        if head_type == "fully_convolutional":
            self.classifier = nn.Sequential(
                nn.Conv2d(lastconv_output_channels, num_classes, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(num_classes),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        elif head_type == "mlp":
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1),
                nn.Linear(lastconv_output_channels, last_channel),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(last_channel, num_classes),
            )
        else:
            raise NotImplementedError(f"Head '{head_type}' unknown.")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _feature_forward(self, x: Tensor, return_fmaps=False):
        fmaps = []
        x = self.in_c(x)
        if return_fmaps:
            fmaps.append(x)
        for layer in self.layers:
            x = layer(x)
            if return_fmaps:
                fmaps.append(x)
        x = self.out_c(x)
        if return_fmaps:
            fmaps.append(x)
            return x, fmaps
        return x

    def _clf_forward(self, x: Tensor):
        embed = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x).squeeze()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x, embed

    def forward(self, x: Tensor, return_fmaps: bool = False):
        if return_fmaps:
            x, fmaps = self._feature_forward(x, return_fmaps=True)
            x, _ = self._clf_forward(x)
            return x, fmaps
        x = self._feature_forward(x)
        return self._clf_forward(x)

    def update_params(self, epoch):
        for module in self.modules():
            if isinstance(module, DynamicConv):
                module.update_params(epoch)


def _dymn_conf(width_mult=1.0, reduced_tail=False, dilated=False,
               strides=(2, 2, 2, 2), use_dy_blocks="all", **kwargs):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1
    bneck_conf = partial(DynamicInvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(DynamicInvertedResidualConfig.adjust_channels, width_mult=width_mult)
    activations = ["RE"] * 6 + ["HS"] * 9

    if use_dy_blocks == "all":
        use_dy_block = [True] * 15
    elif use_dy_blocks == "replace_se":
        use_dy_block = [False, False, False, True, True, True, False, False, False,
                        False, True, True, True, True, True]
    else:
        raise NotImplementedError(f"Config use_dy_blocks={use_dy_blocks} not implemented.")

    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, use_dy_block[0], activations[0], 1, 1),
        bneck_conf(16, 3, 64, 24, use_dy_block[1], activations[1], strides[0], 1),
        bneck_conf(24, 3, 72, 24, use_dy_block[2], activations[2], 1, 1),
        bneck_conf(24, 5, 72, 40, use_dy_block[3], activations[3], strides[1], 1),
        bneck_conf(40, 5, 120, 40, use_dy_block[4], activations[4], 1, 1),
        bneck_conf(40, 5, 120, 40, use_dy_block[5], activations[5], 1, 1),
        bneck_conf(40, 3, 240, 80, use_dy_block[6], activations[6], strides[2], 1),
        bneck_conf(80, 3, 200, 80, use_dy_block[7], activations[7], 1, 1),
        bneck_conf(80, 3, 184, 80, use_dy_block[8], activations[8], 1, 1),
        bneck_conf(80, 3, 184, 80, use_dy_block[9], activations[9], 1, 1),
        bneck_conf(80, 3, 480, 112, use_dy_block[10], activations[10], 1, 1),
        bneck_conf(112, 3, 672, 112, use_dy_block[11], activations[11], 1, 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, use_dy_block[12], activations[12], strides[3], dilation),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider,
                   use_dy_block[13], activations[13], 1, dilation),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider,
                   use_dy_block[14], activations[14], 1, dilation),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)
    return inverted_residual_setting, last_channel


def _dymn(inverted_residual_setting, last_channel, pretrained_name, **kwargs):
    model = DyMN(inverted_residual_setting, last_channel, **kwargs)
    if pretrained_name:
        url = pretrained_models.get(pretrained_name)
        if url is None:
            raise NotImplementedError(
                f"pretrained_name='{pretrained_name}' 不在预训练模型列表中。"
                f"可用选项: {list(pretrained_models.keys())}"
            )
        state_dict = load_state_dict_from_url(url, model_dir=model_dir, map_location="cpu")
        try:
            cls_in_sd = state_dict['classifier.5.weight'].shape[0]
            cls_in_model = model.classifier[5].out_features
            if cls_in_sd != cls_in_model:
                print(f"Classes mismatch ({cls_in_sd} vs {cls_in_model}). Dropping final FC layer.")
                del state_dict['classifier.5.weight']
                del state_dict['classifier.5.bias']
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict)
        except (KeyError, IndexError):
            model.load_state_dict(state_dict, strict=False)
    return model


def get_model(num_classes: int = 527, pretrained_name: str = None, width_mult: float = 1.0,
              strides: Tuple[int, int, int, int] = (2, 2, 2, 2),
              context_ratio: int = 4, max_context_size: int = 128, min_context_size: int = 32,
              dyrelu_k: int = 2, no_dyrelu: bool = False,
              dyconv_k: int = 4, no_dyconv: bool = False,
              T_max: float = 30.0, T0_slope: float = 1.0, T1_slope: float = 0.02,
              T_min: float = 1, pretrain_final_temp: float = 1.0,
              no_ca: bool = False, use_dy_blocks="all"):
    if pretrained_name:
        T_max = pretrain_final_temp
    temp_schedule = (T_max, T_min, T0_slope, T1_slope)
    inverted_residual_setting, last_channel = _dymn_conf(
        width_mult=width_mult, strides=strides, use_dy_blocks=use_dy_blocks
    )
    m = _dymn(
        inverted_residual_setting, last_channel, pretrained_name,
        num_classes=num_classes, width_mult=width_mult, strides=strides,
        context_ratio=context_ratio, max_context_size=max_context_size,
        min_context_size=min_context_size, dyrelu_k=dyrelu_k, dyconv_k=dyconv_k,
        no_dyrelu=no_dyrelu, no_dyconv=no_dyconv, no_ca=no_ca,
        temp_schedule=temp_schedule, use_dy_blocks=use_dy_blocks
    )
    return m
