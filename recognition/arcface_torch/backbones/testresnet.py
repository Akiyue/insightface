import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import cv2
import mediapipe as mp
import numpy as np
from torch.utils.checkpoint import checkpoint

class FaceMaskExtractor:
    def __init__(self, image_size=(112, 112)):
        self.detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.image_size = image_size

    def get_face_mask(self, image_tensor):
        """
        Args:
            image_tensor: torch.Tensor with shape [B, 3, H, W] and pixel values in [0, 1]
        Returns:
            mask_tensor: torch.Tensor with shape [B, 1, H, W] with 1s in face regions
        """
        image_tensor = image_tensor.detach().cpu()
        b, c, h, w = image_tensor.shape
        masks = []

        for i in range(b):
            img_np = (image_tensor[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            results = self.detector.process(img_np)

            mask = np.zeros((h, w), dtype=np.float32)
            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    mask[y1:y2, x1:x2] = 1.0
            masks.append(torch.tensor(mask).unsqueeze(0))

        return torch.stack(masks).to(image_tensor.device) 

class CustomInputPreprocessor(nn.Module):
    def __init__(self, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std
        self.mask_extractor = FaceMaskExtractor()

    def forward(self, rgb_images):
        grayscale = TF.rgb_to_grayscale(rgb_images)  # [B, 1, H, W]
        mask = self.mask_extractor.get_face_mask(rgb_images)  # [B, 1, H, W]

        noise = torch.randn_like(grayscale) * self.noise_std
        noisy = torch.clamp(grayscale + noise, 0.0, 1.0)

        return torch.cat([grayscale, mask, noisy], dim=1)  # [B, 3, H, W]

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample

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
        return checkpoint(self.forward_impl, x) if self.training else self.forward_impl(x)

class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self, block, layers, input_channels=3, num_features=512, dropout=0.4, use_preprocessing=True, noise_std=0.1):
        super().__init__()
        self.use_preprocessing = use_preprocessing
        if use_preprocessing:
            self.preprocessor = CustomInputPreprocessor(noise_std=noise_std)
            input_channels = 3

        self.inplanes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-5),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.use_preprocessing:
            x = self.preprocessor(x)

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
        x = self.fc(x)
        x = self.features(x)
        return x

