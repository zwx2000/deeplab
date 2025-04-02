from nets.xception import xception
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.mobilenetv2 import mobilenetv2
from nets.resnet101 import ResNet101
from nets.mobilenetv3 import mobilenetv3

class XceptionNet(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(XceptionNet, self).__init__()
        from functools import partial
        
        model           = mobilenetv2(pretrained)
        self.features   = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:2](x)
        x = self.features[2:](low_level_features)
        return low_level_features, x 


#-----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
#-----------------------------------------#
class ASPP(nn.Module):
    def __init__(self, dim_in=24, dim_out=24, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.dim_in = 24  # 假设 dim_in 是 24，根据实际情况调整
        self.dim_out = dim_out

        self.branch1 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=False),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, 1, padding=4*rate, dilation=4*rate, bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, 1, padding=8*rate, dilation=8*rate, bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, 1, padding=16*rate, dilation=16*rate, bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )
        self.branch6_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
        self.branch6_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch6_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
                nn.Conv2d(dim_out*6, dim_out, 1, 1, padding=0,bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )

    def forward(self, x):
        print(f"Input size: {x.size()}")  # 打印输入的尺寸，帮助调试
        if len(x.size()) != 4:
            # 如果输入不是4维的，调整为适当的形状
            # 例如，如果是3维的，可以增加批次维度
            x = x.unsqueeze(0)  # 增加一个批次维度，或根据需要调整
            x = x.unsqueeze(2)  # 添加高度维度，形状变为 [1, channels, 1, features]
            x = x.unsqueeze(3)  # 添加宽度维度，形状变为 [1, channels, 1, features]

        [b, c, height, width] = x.size()  # 确保x是四维张量后解包
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        conv3x3_4 = self.branch5(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#

        #-----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #-----------------------------------------#
        global_feature = torch.mean(x,2,True)
        global_feature = torch.mean(global_feature,3,True)
        global_feature = self.branch6_conv(global_feature)
        global_feature = self.branch6_bn(global_feature)
        global_feature = self.branch6_relu(global_feature)
        global_feature = F.interpolate(global_feature, (height, width), None, 'bilinear', True)

        #-----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        #-----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, conv3x3_4, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


        self.conversion_layer = nn.Conv2d(1000, 24, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        print(f"Input size: {x.size()}")  # 打印输入的尺寸，帮助调试
        if len(x.size()) != 4:
            # 如果输入不是4维的，调整为适当的形状
            # 例如，如果是3维的，可以增加批次维度
            x = x.unsqueeze(0)
            x = x.unsqueeze(2)
            x = x.unsqueeze(3)

        x = self.conversion_layer(x)
        [b, c, height, width] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        conv3x3_4 = self.branch5(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#

        #-----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #-----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch6_conv(global_feature)
        global_feature = self.branch6_bn(global_feature)
        global_feature = self.branch6_relu(global_feature)
        global_feature = F.interpolate(global_feature, (height, width), None, 'bilinear', True)

        #-----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        #-----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, conv3x3_4, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result
class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="MobileNetV2", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone=="resnet101":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            #----------------------------------#
            self.backbone = ResNet101(BatchNorm=nn.BatchNorm2d,downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 256
            low_level_channels = 2048
        elif backbone=="xception":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            #----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone=="mobilenetv2":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            #----------------------------------#
            self.backbone = mobilenetv2(pretrained=pretrained, downsample_factor=downsample_factor,)
            in_channels = 24
            low_level_channels = 24
        elif backbone == "mobilenetv3":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            # ----------------------------------#
            self.backbone = mobilenetv3(pretrained=pretrained)
            in_channels = 24
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        #-----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        #-----------------------------------------#
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
        #----------------------------------#
        #   浅层特征边
        #----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )		

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        #-----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        #-----------------------------------------#
        low_level_features, x = self.backbone(x)

        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)
        
        #-----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        #-----------------------------------------#
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

