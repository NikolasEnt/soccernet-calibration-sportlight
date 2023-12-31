"""Based on code from https://github.com/HRNet/HRNet-Semantic-Segmentation/.
"""

import logging
import os
from typing import List, Tuple

import numpy as np
import torch
import torch._utils
import torch.nn as nn
import torch.nn.functional as F
from src.models.hrnet.hrnet import BasicBlock, Bottleneck, HighResolutionModule

BN_MOMENTUM = 0.1
ALIGN_CORNERS = True

relu_inplace = True
BatchNorm2d = torch.nn.SyncBatchNorm

logger = logging.getLogger(__name__)


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):
    def __init__(self, config):
        super(HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, config.stem_width,
                               kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(config.stem_width, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(config.stem_width, config.stem_width,
                               kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(config.stem_width, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)

        self.stage1_cfg = config.stage1
        num_channels = self.stage1_cfg.num_channels[0]
        block = blocks_dict[self.stage1_cfg.block_type]
        num_blocks = self.stage1_cfg.num_blocks[0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = config.stage2
        num_channels = self.stage2_cfg.num_channels
        block = blocks_dict[self.stage2_cfg.block_type]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = config.stage3
        num_channels = self.stage3_cfg.num_channels
        block = blocks_dict[self.stage3_cfg.block_type]
        num_channels = [
            num_channels[i] * block.expansion for i in range(
                len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = config.stage4
        num_channels = self.stage4_cfg.num_channels
        block = blocks_dict[self.stage4_cfg.block_type]
        num_channels = [
            num_channels[i] * block.expansion for i in range(
                len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        # Numper of conv filters in the last block of the encoder
        self.last_inp_channels = int(np.sum(pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.last_inp_channels,
                out_channels=self.last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(self.last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=self.last_inp_channels,
                out_channels=config.num_classes,
                kernel_size=config.final_conv_kernel,
                stride=1,
                padding=1 if config.final_conv_kernel == 3 else 0),
            nn.Softmax(dim=1)
        )
        self.init_weights(config.pretrain)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config.num_modules
        num_branches = layer_config.num_branches
        num_blocks = layer_config.num_blocks
        num_channels = layer_config.num_channels
        block = blocks_dict[layer_config.block_type]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor],
                                                torch.Tensor]:
        """HRNet backbone

        Args:
            x (torch.Tensor): Input tensor (B, 3, H, W).

        Returns:
            List[torch.Tensor]: A list of one tensor, containing the final
                prediction (B, num_classes, H/4, W/4).
            torch.Tensor: The internal feature tensor
                (B, self.last_inp_channels, H/4, W/4).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg.num_branches):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg.num_branches):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg.num_branches:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg.num_branches):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg.num_branches:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=ALIGN_CORNERS)

        x = torch.cat([x[0], x1, x2, x3], 1)

        x_final = self.last_layer(x)

        return [x_final], x

    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     nn.init.normal_(m.weight, std=0.001)
            if isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if (k in model_dict.keys()
                                   and model_dict[k].shape
                                   == pretrained_dict[k].shape)}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
