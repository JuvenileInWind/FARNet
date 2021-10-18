from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models


class wblock(nn.Module):
    def __init__(self, conv3or1, in_channel, out_channel, y_channel=0):
        super(wblock, self).__init__()
        self.conv3or1 = conv3or1
        self.conv11 = nn.Sequential(OrderedDict([
            ('conv11_1', nn.Conv2d(in_channel + y_channel, out_channel, kernel_size=1)),
            ('norm11_1', nn.BatchNorm2d(out_channel)),
            ('relu11_1', nn.ReLU(inplace=True))
        ]))
        self.conv33 = nn.Sequential(OrderedDict([
            ('conv11_1', nn.Conv2d(in_channel + y_channel, out_channel, kernel_size=3, stride=1, padding=1)),
            ('norm11_1', nn.BatchNorm2d(out_channel)),
            ('relu11_1', nn.ReLU(inplace=True))
        ]))

    def forward(self, x, y):
        x = torch.cat((x, y), 1)
        if self.conv3or1 == 1:
            x = self.conv11(x)
        elif self.conv3or1 == 3:
            x = self.conv33(x)
        return x


class Farnet(nn.Module):
    def __init__(self):
        super(Farnet, self).__init__()
        self.features = models.densenet121(pretrained=True).features

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear')

        self.wblock1 = wblock(1, 512, 512, 256)
        self.wblock2 = wblock(1, 1024, 1024, 256)
        self.wblock3 = wblock(3, 2048, 1024, 256)

        self.w1_conv11_0 = nn.Sequential(OrderedDict([
            ('conv11_0', nn.Conv2d(3, 32, kernel_size=1)),
            ('norm11_0', nn.BatchNorm2d(32)),
            ('relu11_0', nn.ReLU(inplace=True)),
        ]))
        self.w1_conv33_01 = nn.Sequential(OrderedDict([
            ('conv11_1', nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)),
            ('norm11_1', nn.BatchNorm2d(512)),
            ('relu11_1', nn.ReLU(inplace=True)),
        ]))
        self.w1_conv11_1 = nn.Sequential(OrderedDict([
            ('conv11_1', nn.Conv2d(512, 256, kernel_size=1)),
            ('norm11_1', nn.BatchNorm2d(256)),
            ('relu11_1', nn.ReLU(inplace=True)),
        ]))

        self.w1_conv11_2 = nn.Sequential(OrderedDict([
            ('conv11_2', nn.Conv2d(1280, 256, kernel_size=1)),
            ('norm11_2', nn.BatchNorm2d(256)),
            ('relu11_2', nn.ReLU(inplace=True)),
        ]))
        self.w1_conv11_3 = nn.Sequential(OrderedDict([
            ('conv11_3', nn.Conv2d(768, 256, kernel_size=1)),
            ('norm11_3', nn.BatchNorm2d(256)),
            ('relu11_3', nn.ReLU(inplace=True)),
        ]))

        self.mid_conv11_1 = nn.Sequential(OrderedDict([
            ('conv11_4', nn.Conv2d(512, 256, kernel_size=1)),
            ('norm11_4', nn.BatchNorm2d(256)),
            ('relu11_4', nn.ReLU(inplace=True)),
        ]))

        self.w2_conv11_1 = nn.Sequential(OrderedDict([
            ('conv11_1', nn.Conv2d(1024, 256, kernel_size=1)),
            ('norm11_1', nn.BatchNorm2d(256)),
            ('relu11_1', nn.ReLU(inplace=True)),
        ]))
        self.w2_conv11_2 = nn.Sequential(OrderedDict([
            ('conv11_2', nn.Conv2d(1280, 256, kernel_size=1)),
            ('norm11_2', nn.BatchNorm2d(256)),
            ('relu11_2', nn.ReLU(inplace=True)),
        ]))
        self.w2_conv11_3 = nn.Sequential(OrderedDict([
            ('conv11_3', nn.Conv2d(768, 256, kernel_size=1)),
            ('norm11_3', nn.BatchNorm2d(256)),
            ('relu11_3', nn.ReLU(inplace=True)),
        ]))
        self.w2_conv11_4 = nn.Sequential(OrderedDict([
            ('conv11_4', nn.Conv2d(512, 128, kernel_size=1)),
            ('norm11_4', nn.BatchNorm2d(128)),
            ('relu11_4', nn.ReLU(inplace=True)),
        ]))
        self.w2_conv11_5 = nn.Sequential(OrderedDict([
            ('conv11_4', nn.Conv2d(192, 64, kernel_size=1)),
            ('norm11_4', nn.BatchNorm2d(64)),
            ('relu11_4', nn.ReLU(inplace=True)),
        ]))
        self.conv33_stride1 = nn.Sequential(OrderedDict([
            ('conv33_0', nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)),
            ('norm33_0', nn.BatchNorm2d(512)),
            ('relu33_0', nn.ReLU(inplace=True)),
        ]))
        self.conv33_stride2 = nn.Sequential(OrderedDict([
            ('conv33_0', nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)),
            ('norm33_0', nn.BatchNorm2d(1024)),
            ('relu33_0', nn.ReLU(inplace=True)),
        ]))
        self.conv33_stride3 = nn.Sequential(OrderedDict([
            ('conv33_0', nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1)),
            ('norm33_0', nn.BatchNorm2d(2048)),
            ('relu33_0', nn.ReLU(inplace=True)),
        ]))
        self.conv_33_refine1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_33_refine2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_11_refine = nn.Conv2d(64, 19, kernel_size=1)
        self.conv_33_last1 = nn.Conv2d(115, 115, kernel_size=3, stride=1, padding=1)
        self.conv_33_last2 = nn.Conv2d(115, 115, kernel_size=3, stride=1, padding=1)
        self.conv_11_last = nn.Conv2d(115, 19, kernel_size=1)

    def forward(self, x):
        w1_f0 = self.w1_conv11_0(x)
        x = self.features[0](x)
        w1_f1 = x
        for i in range(1, 5):
            x = self.features[i](x)
        w1_f2 = x
        for i in range(5, 7):
            x = self.features[i](x)
        w1_f3 = x
        for i in range(7, 9):
            x = self.features[i](x)
        w1_f4 = x
        for i in range(9, 12):
            x = self.features[i](x)
        # first upsample and concat
        x = self.w1_conv33_01(x)
        x = self.w1_conv11_1(x)
        w2_f5 = x
        x = self.upsample2(x)
        x = torch.cat((x, w1_f4), 1)
        # second upsample and concat
        x = self.w1_conv11_2(x)
        w2_f4 = x
        x = self.upsample2(x)
        x = torch.cat((x, w1_f3), 1)
        # third upsample and concat
        x = self.w1_conv11_3(x)
        w2_f3 = x
        x = self.upsample2(x)
        x = torch.cat((x, w1_f2), 1)

        x = self.mid_conv11_1(x)
        w3_f2 = x

        x = self.conv33_stride1(x)
        x = self.wblock1(x, w2_f3)
        w3_f3 = x
        x = self.conv33_stride2(x)
        x = self.wblock2(x, w2_f4)
        w3_f4 = x
        x = self.conv33_stride3(x)
        x = self.wblock3(x, w2_f5)
        x = self.w2_conv11_1(x)
        x = self.upsample2(x)
        x = torch.cat((x, w3_f4), 1)
        x = self.w2_conv11_2(x)
        x = self.upsample2(x)
        x = torch.cat((x, w3_f3), 1)
        x = self.w2_conv11_3(x)
        x = self.upsample2(x)
        x = torch.cat((x, w3_f2), 1)
        x = self.w2_conv11_4(x)
        x = self.upsample2(x)
        x = torch.cat((x, w1_f1), 1)
        x = self.w2_conv11_5(x)

        refine_hp = self.conv_33_refine1(x)
        refine_hp = self.conv_11_refine(refine_hp)

        x = self.upsample2(x)
        refine1_up = self.upsample2(refine_hp)
        x = torch.cat((x, w1_f0, refine1_up), 1)

        # output
        hp = self.conv_33_last1(x)
        hp = self.conv_11_last(hp)
        return hp, refine_hp
