import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
import torchvision.transforms.functional as F

__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200', 'CustomInputPreprocessor']
using_ckpt = False


# ==========================================================================================
# LỚP TIỀN XỬ LÝ MỚI
# ==========================================================================================
class CustomInputPreprocessor(nn.Module):
    """
    Module để tiền xử lý ảnh đầu vào.
    Chuyển đổi ảnh RGB (3 kênh) thành tensor 3 kênh mới:
    - Kênh 1: Ảnh xám (Grayscale)
    - Kênh 2: Mask khuôn mặt (được cung cấp)
    - Kênh 3: Ảnh xám cộng với nhiễu Gaussian
    """
    def __init__(self, noise_std=0.1):
        """
        Khởi tạo preprocessor.
        :param noise_std: Độ lệch chuẩn của nhiễu Gaussian được thêm vào.
        """
        super().__init__()
        self.noise_std = noise_std

    def forward(self, rgb_images, face_masks):
        """
        :param rgb_images: Tensor ảnh RGB với shape [B, 3, H, W]
        :param face_masks: Tensor mask khuôn mặt với shape [B, 1, H, W]
        :return: Tensor đã xử lý với shape [B, 3, H, W]
        """
        # Kênh 1: Chuyển đổi sang ảnh xám
        grayscale_channel = F.rgb_to_grayscale(rgb_images)

        # Kênh 2: Mask khuôn mặt
        mask_channel = face_masks

        # Kênh 3: Ảnh xám cộng nhiễu
        noise = torch.randn_like(grayscale_channel) * self.noise_std
        noisy_grayscale_channel = grayscale_channel + noise
        noisy_grayscale_channel = torch.clamp(noisy_grayscale_channel, 0.0, 1.0) # Giả sử ảnh đầu vào đã được chuẩn hóa về [0,1]

        # Kết hợp 3 kênh
        processed_input = torch.cat([grayscale_channel, mask_channel, noisy_grayscale_channel], dim=1)
        return processed_input

# ==========================================================================================
# CÁC LỚP CÒN LẠI CỦA MÔ HÌNH
# ==========================================================================================

# LỚP FourierFeatureMapping ĐÃ BỊ XÓA BỎ

# FAN Layer
class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim=None, p_ratio=0.25, activation='gelu', use_p_bias=True):
        super(FANLayer, self).__init__()
        assert 0 < p_ratio < 0.5, "p_ratio must be in (0, 0.5)"
        self.p_ratio = p_ratio
        self.output_dim = output_dim if output_dim is not None else input_dim
        p_dim = int(self.output_dim * p_ratio)
        g_dim = self.output_dim - 2 * p_dim
        self.linear_p = nn.Linear(input_dim, p_dim, bias=use_p_bias)
        self.linear_g = nn.Linear(input_dim, g_dim)
        if isinstance(activation, str):
            self.activation = getattr(nn.functional, activation)
        else:
            self.activation = activation if activation else lambda x: x

    def forward(self, x):
        p = self.linear_p(x)
        g = self.activation(self.linear_g(x))
        return torch.cat([torch.cos(p), torch.sin(p), g], dim=-1)

# Conv helpers
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
                     groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# Basic block
class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward_impl(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

    def forward(self, x):
        if self.training and using_ckpt:
            return checkpoint(self.forward_impl, x)
        else:
            return self.forward_impl(x)

# Main ResNet
class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self, block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False,
                 use_fan=True): # << THAY ĐỔI: Đã xóa tham số `use_ffm`
        super(IResNet, self).__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        # self.use_ffm = use_ffm # << THAY ĐỔI: Đã xóa
        self.use_fan = use_fan

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")

        self.groups = groups
        self.base_width = width_per_group

        # << THAY ĐỔI: Logic của conv1 được đơn giản hóa, luôn nhận 3 kênh đầu vào
        # Đầu vào bây giờ luôn là 3 kênh (ảnh xám, mask, ảnh xám nhiễu)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

        fc_in = 512 * block.expansion * self.fc_scale
        self.fc = nn.Linear(fc_in, num_features)
        if self.use_fan:
            self.fan = FANLayer(num_features, output_dim=num_features, p_ratio=0.25)

        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        # << THAY ĐỔI: Đầu vào x bây giờ được kỳ vọng là tensor đã qua xử lý
        with torch.cuda.amp.autocast(self.fp16):
            # << THAY ĐỔI: Dòng gọi FFM đã bị xóa
            # if self.use_ffm:
            #     x = self.ffm(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        if self.use_fan:
            x = self.fan(x)
        x = self.features(x)
        return x

def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    # << THAY ĐỔI: Loại bỏ `use_ffm` khỏi kwargs để tránh lỗi
    kwargs.pop('use_ffm', None)
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError("Pretrained not supported")
    return model

def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained, progress, **kwargs)

def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained, progress, **kwargs)

def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained, progress, **kwargs)
