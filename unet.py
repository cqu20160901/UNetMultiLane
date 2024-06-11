import torch
import torch.nn as nn
from torch.nn import init


def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


class Upsample(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(Upsample, self).__init__()
        self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_size, out_size, 1))

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)


class LineType(nn.Module):
    def __init__(self, in_size, lane_num, type_num):
        super(LineType, self).__init__()
        self.lane_num = lane_num
        self.type_num = type_num

        # 30 * 40
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(64, eps=1e-5, momentum=0.01, affine=True),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(64, eps=1e-5, momentum=0.01, affine=True),
                                   nn.ReLU(inplace=True))

        # 15 * 20
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(128, eps=1e-5, momentum=0.01, affine=True),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(128, eps=1e-5, momentum=0.01, affine=True),
                                   nn.ReLU(inplace=True))
        # 8 * 10
        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(256, eps=1e-5, momentum=0.01, affine=True),
                                   nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(256, eps=1e-5, momentum=0.01, affine=True),
                                   nn.ReLU(inplace=True))
        # 4 * 5
        self.conv7 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(256, eps=1e-5, momentum=0.01, affine=True),
                                   nn.ReLU(inplace=True))

        self.conv8 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(256, eps=1e-5, momentum=0.01, affine=True),
                                   nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(256, eps=1e-5, momentum=0.01, affine=True),
                                   nn.ReLU(inplace=True))

        self.classfier = nn.Sequential(
            nn.Linear(256 * 4 * 5, 64),
            nn.Linear(64, self.lane_num * self.type_num)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv9(x)
        x = x.view(x.size()[0], -1)
        x = self.classfier(x)
        x = x.view(-1, self.lane_num, self.type_num)
        return x


# 模型输入以640x480为例（其它输入分辨率自行适配）
class UNetMultiLane(nn.Module):
    def __init__(self, in_channels=1, lane_num=9, type_num=11, width_mult=0.5, is_deconv=True, is_batchnorm=True, is_ds=True):
        super(UNetMultiLane, self).__init__()
        self.in_channels = in_channels
        self.width_mult = width_mult
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x * self.width_mult) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat01 = Upsample(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = Upsample(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = Upsample(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = Upsample(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = Upsample(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = Upsample(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = Upsample(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = Upsample(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = Upsample(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = Upsample(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], lane_num, 1)
        self.final_2 = nn.Conv2d(filters[0], lane_num, 1)
        self.final_3 = nn.Conv2d(filters[0], lane_num, 1)
        self.final_4 = nn.Conv2d(filters[0], lane_num, 1)

        self.LineType = LineType(in_size=filters[4], lane_num=lane_num - 1, type_num=type_num)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)     # 16*480*640
        maxpool0 = self.maxpool(X_00)  # 16*240*320
        X_10 = self.conv10(maxpool0)   # 32*240*320
        maxpool1 = self.maxpool(X_10)  # 32*120*160
        X_20 = self.conv20(maxpool1)   # 64*120*160
        maxpool2 = self.maxpool(X_20)  # 64*60*80
        X_30 = self.conv30(maxpool2)   # 128*60*80
        maxpool3 = self.maxpool(X_30)  # 128*30*40
        X_40 = self.conv40(maxpool3)   # 256*30*40

        lane_type = self.LineType(X_40)

        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)

        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)

        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)

        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1 + final_2 + final_3 + final_4)

        if self.is_ds:
            return [final, lane_type]
        else:
            return [final_4, lane_type]


if __name__ == '__main__':
    print('This is main ...')
    model = UNetMultiLane(in_channels=3, lane_num=9, type_num=11, width_mult=0.0625)
    img = torch.ones(size=(1, 3, 480, 640))
    seg, lane_type = model(img)
    print(seg.shape, lane_type.shape)