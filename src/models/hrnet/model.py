import torch
import torch.nn.functional as F
from torch import nn

from src.models.hrnet.hrnet import HighResolutionNet


def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True,
         dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels,
                         kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


def conv_dw(in_channels, out_channels, kernel_size=3, padding=1, stride=1,
            dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                  padding, dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels),
            conv(num_channels, num_channels),
            conv(num_channels, num_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1,
                 padding=0, bn=False, relu=False)
        )
        self.final = nn.Softmax(dim=1)

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.final(self.heatmaps(trunk_features))
        return [heatmaps]


class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels,
                            kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )
        self.final = nn.Softmax(dim=1)

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return self.final(initial_features + trunk_features)


class UShapedContextBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder1 = nn.Sequential(
            conv(in_channels, in_channels*2, stride=2),
            conv(in_channels*2, in_channels*2),
        )
        self.encoder2 = nn.Sequential(
            conv(in_channels*2, in_channels*2, stride=2),
            conv(in_channels*2, in_channels*2),
        )
        self.decoder2 = nn.Sequential(
            conv(in_channels*2 + in_channels*2, in_channels*2),
            conv(in_channels*2, in_channels*2),
        )
        self.decoder1 = nn.Sequential(
            conv(in_channels*3, in_channels*2),
            conv(in_channels*2, in_channels)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)

        size_e1 = (e1.size()[2], e1.size()[3])
        size_x = (x.size()[2], x.size()[3])
        d2 = self.decoder2(
            torch.cat([e1, F.interpolate(e2, size=size_e1, mode='bilinear',
                                         align_corners=False)], 1))
        d1 = self.decoder1(
            torch.cat([x, F.interpolate(d2, size=size_x, mode='bilinear',
                                        align_corners=False)], 1))

        return d1


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps):
        super().__init__()

        self.trunk = nn.Sequential(
            UShapedContextBlock(in_channels),
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1,
                 padding=0, bn=False, relu=False),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        return [heatmaps]


class HRNetHeatmap(nn.Module):
    def __init__(self, hrnet_config, num_refinement_stages: int = 0,
                 num_heatmaps: int = 38):
        super().__init__()
        self.model = HighResolutionNet(hrnet_config)

        self.refinement_stages = nn.ModuleList()
        for _ in range(num_refinement_stages):
            self.refinement_stages.append(
                RefinementStage(self.model.last_inp_channels + num_heatmaps,
                                self.model.last_inp_channels,
                                num_heatmaps))

    def forward(self, x):
        stages_output, x_encoder = self.model(x)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat(
                    [x_encoder, stages_output[-1]], dim=1)))

        return stages_output
