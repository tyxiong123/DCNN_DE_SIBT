import torch
import torch.nn as nn
import torch.nn.functional as F


def upkern_init_load_weights ( network , pretrained_net ):
    pretrn_dct = pretrained_net . state_dict ()
    model_dct = network . state_dict ()
    
    for k in model_dct . keys ():

        if k in model_dct . keys () and k in pretrn_dct . keys ():

            spt_dims1 = model_dct[k].shape
            spt_dims2 = pretrn_dct[k].shape
            if spt_dims1 == spt_dims2 : # standard init
                model_dct[k] = pretrn_dct[k]
            else : # Upsampled kernel init
                inc1 , outc1 , * spt_dims1 = model_dct[k].shape
                model_dct[k] = F.interpolate (pretrn_dct[k],size = spt_dims1 ,mode ='trilinear')
    network.load_state_dict (model_dct)
    return network


    
class Up(nn.Module):
    def __init__(self, down_in_channels, in_channels, out_channels, conv_block, interpolation=True, net_mode='3d'):
        super(Up, self).__init__()

        if net_mode == '2d':
            inter_mode = 'bilinear'
            trans_conv = nn.ConvTranspose2d
        elif net_mode == '3d':
            inter_mode = 'trilinear'
            trans_conv = nn.ConvTranspose3d
        else:
            inter_mode = None
            trans_conv = None

        if interpolation == True:
            self.up = nn.Upsample(scale_factor=2, mode=inter_mode, align_corners=True)
        else:
            self.up = trans_conv(down_in_channels, down_in_channels, 2, stride=2)

        self.conv = conv_block(in_channels + down_in_channels, out_channels, net_mode=net_mode)
        self.Att = Attention_block(F_g = down_in_channels, F_l = in_channels, F_int = round(in_channels/2))

    def forward(self, down_x, x):
        up_x = self.up(down_x)

        x = self.Att(g = up_x, x = x)
        x = torch.cat((up_x, x), dim=1)

        x = self.conv(x)

        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block, net_mode='3d'):
        super(Down, self).__init__()
        if net_mode == '2d':
            maxpool = nn.MaxPool2d
        elif net_mode == '3d':
            maxpool = nn.MaxPool3d
        else:
            maxpool = None

        self.conv = conv_block(in_channels, out_channels, net_mode=net_mode)

        self.down = maxpool(2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        out = self.down(x)

        return x, out



class standard_double_conv_bigkernel(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normalization=True, kernel_size=3, net_mode='3d'):
        super(standard_double_conv_bigkernel, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            conv = None
            bn = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bach_normalization = batch_normalization

        #self.conv_res = conv(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        if self.in_channels<self.out_channels:
            self.conv_1 = conv(self.in_channels, self.in_channels, kernel_size=5, stride=1, padding=2)
            self.bn_1 = bn(self.in_channels)
            self.conv_2 = conv(self.in_channels, self.out_channels, kernel_size=5, stride=1, padding=2)
        else:
            self.conv_1 = conv(self.in_channels, self.out_channels, kernel_size=5, stride=1, padding=2)
            self.bn_1 = bn(self.out_channels)
            self.conv_2 = conv(self.out_channels, self.out_channels, kernel_size=5, stride=1, padding=2)          
        self.bn_2 = bn(self.out_channels)

        

    def forward(self, input):

        #y = self.conv_res(input)
        x = self.conv_1(input)
        x = self.bn_1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = nn.LeakyReLU(0.2)(x)

        #x+= y

        return x

class standard_double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normalization=True, kernel_size=3, net_mode='3d'):
        super(standard_double_conv, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            conv = None
            bn = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bach_normalization = batch_normalization

        #self.conv_res = conv(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        if self.in_channels<self.out_channels:
            self.conv_1 = conv(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1)
            self.bn_1 = bn(self.in_channels)
            self.conv_2 = conv(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv_1 = conv(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
            self.bn_1 = bn(self.out_channels)
            self.conv_2 = conv(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)          
        self.bn_2 = bn(self.out_channels)

        

    def forward(self, input):

        #y = self.conv_res(input)
        x = self.conv_1(input)
        x = self.bn_1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = nn.LeakyReLU(0.2)(x)

        #x+= y

        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int, net_mode='3d'):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class DCNN_DE_LK(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, conv_block=standard_double_conv, net_mode='3d'):
        super(DCNN_DE_LK, self).__init__()
        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d

        self.bn_inc = bn(in_channels)

        self.inc = conv(in_channels, 8, kernel_size=5, stride=1, padding=2)

        # down
        self.down1 = Down(8, filter_num_list[0], conv_block=standard_double_conv_bigkernel, net_mode=net_mode)
        self.down2 = Down(filter_num_list[0], filter_num_list[1], conv_block=standard_double_conv_bigkernel, net_mode=net_mode)
        self.down3 = Down(filter_num_list[1], filter_num_list[2], conv_block=conv_block, net_mode=net_mode)
 

        self.bridge = conv_block(filter_num_list[2], filter_num_list[3], net_mode=net_mode)


        self.up1 = Up(filter_num_list[3], filter_num_list[2], filter_num_list[2], conv_block=conv_block,
                      net_mode=net_mode)
        self.up2 = Up(filter_num_list[2], filter_num_list[1], filter_num_list[1], conv_block=standard_double_conv_bigkernel,
                      net_mode=net_mode)
        self.up3 = Up(filter_num_list[1], filter_num_list[0], filter_num_list[0], conv_block=standard_double_conv_bigkernel,
                      net_mode=net_mode)

 
        self.final_conv_1 = conv(filter_num_list[0], class_num, kernel_size=5, stride=1, padding=2)


    def forward(self, input):
        
        x = input
        y = input[:,0,:,:,:].unsqueeze(1)  # TG-43 Dose
        x = self.bn_inc(x)

        x = self.inc(x)

        conv1, x = self.down1(x)

        conv2, x = self.down2(x)

        conv3, x = self.down3(x)


        x = self.bridge(x)


        x = self.up1(x, conv3)

        x = self.up2(x, conv2)

        x = self.up3(x, conv1)

        x = self.final_conv_1(x)

        x+=y
        x = nn.ReLU()(x)

        return x


class DCNN_DE(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, conv_block=standard_double_conv, net_mode='3d'):
        super(DCNN_DE, self).__init__()
        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d

        self.bn_inc = bn(in_channels)

        self.inc = conv(in_channels, 8, kernel_size=3, stride=1, padding=1)

        # down
        self.down1 = Down(8, filter_num_list[0], conv_block=conv_block, net_mode=net_mode)
        self.down2 = Down(filter_num_list[0], filter_num_list[1], conv_block=conv_block, net_mode=net_mode)
        self.down3 = Down(filter_num_list[1], filter_num_list[2], conv_block=conv_block, net_mode=net_mode)
 

        self.bridge = conv_block(filter_num_list[2], filter_num_list[3], net_mode=net_mode)


        self.up1 = Up(filter_num_list[3], filter_num_list[2], filter_num_list[2], conv_block=conv_block,
                      net_mode=net_mode)
        self.up2 = Up(filter_num_list[2], filter_num_list[1], filter_num_list[1], conv_block=conv_block,
                      net_mode=net_mode)
        self.up3 = Up(filter_num_list[1], filter_num_list[0], filter_num_list[0], conv_block=conv_block,
                      net_mode=net_mode)

 
        self.final_conv_1 = conv(filter_num_list[0], class_num, kernel_size=3, stride=1, padding=1)


    def forward(self, input):
        
        x = input
        y = input[:,0,:,:,:].unsqueeze(1)  # TG-43 Dose
        x = self.bn_inc(x)

        x = self.inc(x)

        conv1, x = self.down1(x)

        conv2, x = self.down2(x)

        conv3, x = self.down3(x)


        x = self.bridge(x)


        x = self.up1(x, conv3)

        x = self.up2(x, conv2)

        x = self.up3(x, conv1)

        x = self.final_conv_1(x)

        x+=y
        x = nn.ReLU()(x)

        return x









