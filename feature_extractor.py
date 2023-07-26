"""
This script is used for extracting feature representations from person images for data association and re-identification
using learned ReId ResNet34 model.
"""

import cv2
import math
import matplotlib.pyplot as plt

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet34(nn.Module):
    def __init__(self):
        block = BasicBlock
        layers = [3, 4, 6, 3]

        self.inplanes = 64
        super(ResNet34, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feat_bn = nn.BatchNorm2d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.feat_bn(x)
        features = x.view(x.size(0), -1)
        return features


class FeatureExtractor:
    def __init__(self, reid_model_weights='./model/reid_model.pth'):
        self.weights_path = reid_model_weights
        self.model = ResNet34()
        self.model = torch.nn.DataParallel(self.model).to(device)
        model_dict = self.model.state_dict()
        pretrained = torch.load(reid_model_weights)
        pretrained_dict = pretrained['state_dict']
        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # Load the new state dict
        self.model.load_state_dict(model_dict)

        self.model.eval()

    def extract_features_image(self, im):
        input_np = cv2.resize(im, (128, 384))
        input_np = input_np.transpose((2, 0, 1)) / 255.0  # This converts (H,W,C) to (C.H,W) and normalizes it. 				  
        input = torch.from_numpy(input_np)
        input = input.unsqueeze(0).float()
        feats = self.model(input.to(device))
        features = feats.data.cpu().numpy()

        return features

    def extract_features_batch(self, ims):
        if len(ims) > 0:
            crop_batch = torch.Tensor(len(ims), 3, 384, 128)
            for i, crop in enumerate(ims):
                input_np = cv2.resize(crop, (128, 384))
                input_np = input_np.transpose((2, 0, 1)) / 255.0  # This converts (H,W,C) to (C.H,W) and normalizes it.
                inputs = torch.from_numpy(input_np).float()
                crop_batch[i] = inputs

            feats = self.model(crop_batch.to(device))
            features = feats.data.cpu().numpy()

            return features
        return None


def main():
    feature_extractor = FeatureExtractor("./model/reid_model.pth")
    im = plt.imread("./person_crop.jpg")
    feats = feature_extractor.extract_features_image(im)
    print(feats.shape)  # (1, 512)
    

if __name__ == '__main__':
    main()
