import torch.nn as nn
# import torch
# import torch.nn.functional as F
from models.model_utils import *
from tools.Register import Registers



@Registers.runners.register_with_name('Model1')
class Model1(nn.Module):

    def __init__(self, args):
        super().__init__()
        # print('model init', '.'*10)
        self.img_super_resnet = Super_resnet(1, args.base_channels, args.super_resnet_deep, res_scale=args.res_scale1)  # for image
        self.xt_super_resnet = Super_resnet(1, args.base_channels, args.super_resnet_deep, res_scale=args.res_scale1)  # for xt

        
        self.unet = Unet1(args.base_channels, args.unet_rate, res_scale=args.res_scale2) # map_down=args.map_down, 
        # self.hello = 'hello'

    # def cam(self, x, y, t, location):
    #     classify_map = self.classify_model.get_map(y)
    #     x = self.xt_super_resnet(x)
    #     y = self.img_super_resnet(y)
    #     return self.unet.cam(x, y, t, classify_map, location)

    # def feature_map(self, x, y, t, location):
    #     classify_map = self.classify_model.get_map(y)
    #     x = self.xt_super_resnet(x)
    #     y = self.img_super_resnet(y)
    #     return self.unet.feature_map(x, y, t, classify_map, location)


    def forward(self, x, t, y): # x-xt y-image
        # print('model forward', '.'*10)
        # print(x, t, y)
        # print(self.hello)
        # exit()
        # print(x.shape, t, y.shape)
        x = self.xt_super_resnet(x)
        y = self.img_super_resnet(y)
        return self.unet(x, t, y)

    
@Registers.runners.register_with_name('Model2')
class Unet2(nn.Module):
    def __init__(self,args): # , map_down=False
        super().__init__()
        base_channels, rate = args.base_channels, args.unet_rate
        in_ch = args.input_channels
        time_emb_scale=1.0
        num_groups=16
        res_scale=0.2
        self.res_scale = res_scale
        #rate = [1,1,1,2,2,4,4]
        # in_ch = 1
        out_ch = 1
        ch = [base_channels] + [base_channels * i for i in rate]
        use_attention = args.use_attention
        down_mamba = args.down_mamba
        up_mamba = args.up_mamba
        bottleneck_mamba = args.bottleneck_mamba
        assert len(use_attention) == len(rate) == len(down_mamba) == len(up_mamba), f'{use_attention}--{rate}--{down_mamba}--{up_mamba}'

        self.in_block_x1 = DoubleConv(in_ch, base_channels, mid_channels=int(base_channels//2) if int(base_channels//2) > in_ch else None)
        self.in_block_x2 = DoubleConv(in_ch, base_channels, mid_channels=int(base_channels//2) if int(base_channels//2) > in_ch else None)
        self.down_blocks_x1 = nn.ModuleList()
        self.down_blocks_x2 = nn.ModuleList()
        print(f'double conv: in-{in_ch},out-{base_channels}')

        self.attention_blocks = nn.ModuleList()
        for i in range(len(ch)-1):
            ng = num_groups
            while (ng%ch[i+1] != 0 or ng%ch[i+1] != 0) and ng != 1:
                ng//=2
            print(f'down: in-{ch[i]},out-{ch[i+1]}')
            self.down_blocks_x1.append(
                Down_block_mamba(ch[i], ch[i+1], base_channels, time_emb_scale, num_groups=ng)
                if down_mamba[i] else
                Down_block(ch[i], ch[i+1], base_channels, time_emb_scale, num_groups=ng)
                                    ) 
            self.down_blocks_x2.append(
                Down_block_mamba(ch[i], ch[i+1], base_channels, time_emb_scale, num_groups=ng)
                if down_mamba[i] else
                Down_block(ch[i], ch[i+1], base_channels, time_emb_scale, num_groups=ng)
                                    )
            self.attention_blocks.append(self_attention(ch[i+1], reduce=32) if use_attention[i]==True else empty())
        # self.attention_blocks[-1] = nn.Sequential()

        ng = num_groups
        while ng%ch[-1] != 0 and ng != 1:
                ng//=2
        print('bottlenek, ch=', ch[-1])

        self.bottleneck = Bottleneck_mamba(ch[-1], ng) if bottleneck_mamba else Bottleneck(ch[-1], ng)

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(ch)-1)):
            ng = num_groups
            while (ng%ch[i+1] != 0 or ng%ch[i+1] != 0) and ng != 1:
                ng//=2
            print(f'up block: in-{ch[i+1]},out-{ch[i]}')
            self.up_blocks.append(
                Up_block_mamba(ch[i+1], ch[i], base_channels, time_emb_scale,num_groups=ng)
                if up_mamba[i] else
                Up_block(ch[i+1], ch[i], base_channels, time_emb_scale,num_groups=ng)
                )

        self.out_block = nn.Sequential(
            nn.BatchNorm2d(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_ch, 3, padding=1)
        )

    
    def forward(self, x1, t, y):  # x1 for xt, x2 for image
        x1 = self.in_block_x1(x1)
        x2 = self.in_block_x2(y)
        out_x1 = []
        out_x2 = []
        d_x1_, d_x1 = None, x1
        d_x2_, d_x2 = None, x2
        for down_block_x1, down_block_x2, attention_block in zip(self.down_blocks_x1, self.down_blocks_x2, self.attention_blocks):
            d_x1_, d_x1 = down_block_x1(d_x1, t)
            d_x2_, d_x2 = down_block_x2(d_x2, t)
            d_x1, d_x2 = attention_block(d_x1, d_x2)
            out_x1.append(d_x1_)
            out_x2.append(d_x2_)

        u = self.bottleneck(d_x1 + d_x2)


        # u = m + self.down3(resnet_map).mul(self.res_scale)
        # u = m + resnet_map.mul(self.res_scale)
        for up_block, d_x1_, d_x2_ in zip(self.up_blocks, reversed(out_x1), reversed(out_x2)):
            # print('*'*20)
            u = up_block(u, d_x1_+d_x2_, t)

        return self.out_block(u)


    
@Registers.runners.register_with_name('Model3')
class Unet3(nn.Module):
    def __init__(self,args): # , map_down=False
        super().__init__()
        base_channels, rate = args.base_channels, args.unet_rate
        in_ch = args.input_channels
        time_emb_scale=1.0
        num_groups=16
        res_scale=0.2
        self.res_scale = res_scale
        #rate = [1,1,1,2,2,4,4]
        # in_ch = 1
        out_ch = 1
        ch = [base_channels] + [base_channels * i for i in rate]
        use_attention = args.use_attention
        down_mamba = args.down_mamba
        up_mamba = args.up_mamba
        bottleneck_mamba = args.bottleneck_mamba
        assert len(use_attention) == len(rate) == len(down_mamba) == len(up_mamba), f'{use_attention}--{rate}--{down_mamba}--{up_mamba}'

        self.in_block_x1 = DoubleConv(in_ch, base_channels, mid_channels=int(base_channels//2) if int(base_channels//2) > in_ch else None)
        self.in_block_x2 = DoubleConv(in_ch, base_channels, mid_channels=int(base_channels//2) if int(base_channels//2) > in_ch else None)
        self.down_blocks_x1 = nn.ModuleList()
        self.down_blocks_x2 = nn.ModuleList()
        print(f'double conv: in-{in_ch},out-{base_channels}')

        self.attention_blocks = nn.ModuleList()
        for i in range(len(ch)-1):
            ng = num_groups
            while (ng%ch[i+1] != 0 or ng%ch[i+1] != 0) and ng != 1: ng//=2
            print(f'down: in-{ch[i]},out-{ch[i+1]}')
            self.down_blocks_x1.append(
                Down_block_PVM(ch[i], ch[i+1], base_channels, time_emb_scale, num_groups=ng)
                if down_mamba[i] else
                Down_block(ch[i], ch[i+1], base_channels, time_emb_scale, num_groups=ng)
                                    ) 
            self.down_blocks_x2.append(
                Down_block_PVM(ch[i], ch[i+1], base_channels, time_emb_scale, num_groups=ng)
                if down_mamba[i] else
                Down_block(ch[i], ch[i+1], base_channels, time_emb_scale, num_groups=ng)
                                    )
            self.attention_blocks.append(self_attention(ch[i+1], reduce=32) if use_attention[i]==True else empty())
        # self.attention_blocks[-1] = nn.Sequential()

        ng = num_groups
        while ng%ch[-1] != 0 and ng != 1:
                ng//=2
        print('bottlenek, ch=', ch[-1])

        self.bottleneck = Bottleneck_mamba(ch[-1], ng) if bottleneck_mamba else Bottleneck(ch[-1], ng)
        
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(ch)-1)):
            ng = num_groups
            while (ng%ch[i+1] != 0 or ng%ch[i+1] != 0) and ng != 1:
                ng//=2
            print(f'up block: in-{ch[i+1]},out-{ch[i]}')
            self.up_blocks.append(
                Up_block_mamba(ch[i+1], ch[i], base_channels, time_emb_scale,num_groups=ng)
                if up_mamba[i] else
                Up_block(ch[i+1], ch[i], base_channels, time_emb_scale,num_groups=ng)
                )

        self.out_block = nn.Sequential(
            nn.BatchNorm2d(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_ch, 3, padding=1)
        )

    
    def forward(self, x1, t, y):  # x1 for xt, x2 for image
        x1 = self.in_block_x1(x1)
        x2 = self.in_block_x2(y)
        out_x1 = []
        out_x2 = []
        d_x1_, d_x1 = None, x1
        d_x2_, d_x2 = None, x2
        for down_block_x1, down_block_x2, attention_block in zip(self.down_blocks_x1, self.down_blocks_x2, self.attention_blocks):
            d_x1_, d_x1 = down_block_x1(d_x1, t)
            d_x2_, d_x2 = down_block_x2(d_x2, t)
            d_x1, d_x2 = attention_block(d_x1, d_x2)
            out_x1.append(d_x1_)
            out_x2.append(d_x2_)

        u = self.bottleneck(d_x1 + d_x2)


        # u = m + self.down3(resnet_map).mul(self.res_scale)
        # u = m + resnet_map.mul(self.res_scale)
        for up_block, d_x1_, d_x2_ in zip(self.up_blocks, reversed(out_x1), reversed(out_x2)):
            # print('*'*20)
            u = up_block(u, d_x1_+d_x2_, t)

        return self.out_block(u)


@Registers.runners.register_with_name('Model4')
class Unet4(nn.Module):
    def __init__(self,args): # , map_down=False
        super().__init__()
        base_channels, rate = args.base_channels, args.unet_rate
        in_ch = args.input_channels
        time_emb_scale=1.0
        num_groups=16
        res_scale=0.2
        self.res_scale = res_scale
        #rate = [1,1,1,2,2,4,4]
        # in_ch = 1
        out_ch = 1
        ch = [base_channels] + [base_channels * i for i in rate]
        use_attention = args.use_attention
        down_mamba = args.down_mamba
        up_mamba = args.up_mamba
        bottleneck_mamba = args.bottleneck_mamba
        assert len(use_attention) == len(rate) == len(down_mamba) == len(up_mamba), f'{use_attention}--{rate}--{down_mamba}--{up_mamba}'

        self.in_block_x1 = DoubleConv(in_ch, base_channels, mid_channels=int(base_channels//2) if int(base_channels//2) > in_ch else None)
        self.in_block_x2 = DoubleConv(in_ch, base_channels, mid_channels=int(base_channels//2) if int(base_channels//2) > in_ch else None)
        self.down_blocks_x1 = nn.ModuleList()
        self.down_blocks_x2 = nn.ModuleList()
        print(f'double conv: in-{in_ch},out-{base_channels}')

        self.attention_blocks = nn.ModuleList()
        for i in range(len(ch)-1):
            ng = num_groups
            while (ng%ch[i+1] != 0 or ng%ch[i+1] != 0) and ng != 1: ng//=2
            print(f'down: in-{ch[i]},out-{ch[i+1]}')
            self.down_blocks_x1.append(
                Down_block_PVM2(ch[i], ch[i+1], base_channels, time_emb_scale, num_groups=ng)
                if down_mamba[i] else
                Down_block(ch[i], ch[i+1], base_channels, time_emb_scale, num_groups=ng)
                                    ) 
            self.down_blocks_x2.append(
                Down_block_PVM2(ch[i], ch[i+1], base_channels, time_emb_scale, num_groups=ng)
                if down_mamba[i] else
                Down_block(ch[i], ch[i+1], base_channels, time_emb_scale, num_groups=ng)
                                    )
            self.attention_blocks.append(self_attention(ch[i+1], reduce=32) if use_attention[i]==True else empty())
        # self.attention_blocks[-1] = nn.Sequential()

        ng = num_groups
        while ng%ch[-1] != 0 and ng != 1:
                ng//=2
        print('bottlenek, ch=', ch[-1])

        self.bottleneck = Bottleneck_mamba(ch[-1], ng) if bottleneck_mamba else Bottleneck(ch[-1], ng)
        
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(ch)-1)):
            ng = num_groups
            while (ng%ch[i+1] != 0 or ng%ch[i+1] != 0) and ng != 1:
                ng//=2
            print(f'up block: in-{ch[i+1]},out-{ch[i]}')
            self.up_blocks.append(
                Up_block_mamba(ch[i+1], ch[i], base_channels, time_emb_scale,num_groups=ng)
                if up_mamba[i] else
                Up_block(ch[i+1], ch[i], base_channels, time_emb_scale,num_groups=ng)
                )

        self.out_block = nn.Sequential(
            nn.BatchNorm2d(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_ch, 3, padding=1)
        )

    
    def forward(self, x1, t, y):  # x1 for xt, x2 for image
        x1 = self.in_block_x1(x1)
        x2 = self.in_block_x2(y)
        out_x1 = []
        out_x2 = []
        d_x1_, d_x1 = None, x1
        d_x2_, d_x2 = None, x2
        for down_block_x1, down_block_x2, attention_block in zip(self.down_blocks_x1, self.down_blocks_x2, self.attention_blocks):
            d_x1_, d_x1 = down_block_x1(d_x1, t)
            d_x2_, d_x2 = down_block_x2(d_x2, t)
            d_x1, d_x2 = attention_block(d_x1, d_x2)
            out_x1.append(d_x1_)
            out_x2.append(d_x2_)

        u = self.bottleneck(d_x1 + d_x2)


        # u = m + self.down3(resnet_map).mul(self.res_scale)
        # u = m + resnet_map.mul(self.res_scale)
        for up_block, d_x1_, d_x2_ in zip(self.up_blocks, reversed(out_x1), reversed(out_x2)):
            # print('*'*20)
            u = up_block(u, d_x1_+d_x2_, t)

        return self.out_block(u)

@Registers.runners.register_with_name('Model5')
class Unet5(nn.Module):
    def __init__(self,args): # , map_down=False
        super().__init__()
        base_channels, rate = args.base_channels, args.unet_rate
        # in_ch = args.input_channels
        time_emb_scale=1.0
        num_groups=16
        res_scale=0.2
        self.res_scale = res_scale
        #rate = [1,1,1,2,2,4,4]
        # in_ch = 1
        out_ch = 1
        ch = [base_channels] + [base_channels * i for i in rate]
        use_attention = args.use_attention
        down_mamba = args.down_mamba
        up_mamba = args.up_mamba
        mamba_connection = args.mamba_connection
        bottleneck_mamba = args.bottleneck_mamba
        assert len(use_attention) == len(rate) == len(down_mamba) == len(up_mamba) == len(mamba_connection), f'{use_attention}--{rate}--{down_mamba}--{up_mamba}--{mamba_connection}'

        self.in_block_x1 = DoubleConv(args.x_channels, base_channels, mid_channels=int(base_channels//2) if int(base_channels//2) > args.x_channels else None)
        self.in_block_x2 = DoubleConv(args.y_channels, base_channels, mid_channels=int(base_channels//2) if int(base_channels//2) > args.y_channels else None)
        self.down_blocks_x1 = nn.ModuleList()
        self.down_blocks_x2 = nn.ModuleList()
        print(f'double conv: in-{args.x_channels}/{args.y_channels},out-{base_channels}')

        Attention = {
            1: self_attention2,
            2: MambaConnection5,
            3: self_attention3,
            4: self_attention4,
        }
        self.attention_blocks = nn.ModuleList()
        for i in range(len(ch)-1):
            ng = num_groups
            while (ng%ch[i+1] != 0 or ng%ch[i+1] != 0) and ng != 1: ng//=2
            print(f'down: in-{ch[i]},out-{ch[i+1]}')
            self.down_blocks_x1.append(
                Down_block_PVM(ch[i], ch[i+1], base_channels, time_emb_scale, num_groups=ng)
                if down_mamba[i] else
                Down_block(ch[i], ch[i+1], base_channels, time_emb_scale, num_groups=ng)
                                    ) 
            self.down_blocks_x2.append(
                Down_block_PVM(ch[i], ch[i+1], base_channels, time_emb_scale, num_groups=ng)
                if down_mamba[i] else
                Down_block(ch[i], ch[i+1], base_channels, time_emb_scale, num_groups=ng)
                                    )
            self.attention_blocks.append(Attention[use_attention[i]](ch[i+1], reduce=args.reduce) if use_attention[i]>0 else empty())
        # self.attention_blocks[-1] = nn.Sequential()

        ng = num_groups
        while ng%ch[-1] != 0 and ng != 1:
                ng//=2
        print('bottlenek, ch=', ch[-1])

        Merge = {
            0: add,
            1: self_attention5
        }
        self.merge_before_bn = Merge[args.merge](ch[-1], reduce=args.reduce)
        self.bottleneck = Bottleneck_mamba(ch[-1], ng) if bottleneck_mamba else Bottleneck(ch[-1], ng)
        
        self.up_blocks = nn.ModuleList()

        Connection = {
            1: MambaConnection,
            2: MambaConnection2,
            3: MambaConnection3,
            4: MambaConnection4,
            5: self_attention5
        }
        self.connections = nn.ModuleList()
        for i in reversed(range(len(ch)-1)):
            ng = num_groups
            while (ng%ch[i+1] != 0 or ng%ch[i+1] != 0) and ng != 1:
                ng//=2
            print(f'up block: in-{ch[i+1]},out-{ch[i]}')
            self.up_blocks.append(
                Up_block_mamba(ch[i+1], ch[i], base_channels, time_emb_scale,num_groups=ng)
                if up_mamba[i] else
                Up_block(ch[i+1], ch[i], base_channels, time_emb_scale,num_groups=ng)
                )
            self.connections.append(Connection[mamba_connection[i]](ch[i+1]) if mamba_connection[i]>0 else add())

        self.out_block = nn.Sequential(
            nn.BatchNorm2d(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_ch, 3, padding=1)
        )

    
    def forward(self, x1, t, y):  # x1 for xt, x2 for image
        x1 = self.in_block_x1(x1)
        x2 = self.in_block_x2(y)
        out_x1 = []
        out_x2 = []
        d_x1_, d_x1 = None, x1
        d_x2_, d_x2 = None, x2
        for down_block_x1, down_block_x2, attention_block in zip(self.down_blocks_x1, self.down_blocks_x2, self.attention_blocks):
            d_x1_, d_x1 = down_block_x1(d_x1, t)
            d_x2_, d_x2 = down_block_x2(d_x2, t)
            d_x1, d_x2 = attention_block(d_x1, d_x2)
            out_x1.append(d_x1_)
            out_x2.append(d_x2_)

        u = self.bottleneck(self.merge_before_bn(d_x1, d_x2))


        # u = m + self.down3(resnet_map).mul(self.res_scale)
        # u = m + resnet_map.mul(self.res_scale)
        for up_block, connection, d_x1_, d_x2_ in zip(self.up_blocks, self.connections, reversed(out_x1), reversed(out_x2)):
            # print('*'*20)
            u = up_block(u, connection(d_x1_, d_x2_), t)
        return self.out_block(u)



@Registers.runners.register_with_name('Model6')
class Unet6(nn.Module):
    def __init__(self,args): # , map_down=False
        super().__init__()
        base_channels, rate = args.base_channels, args.unet_rate
        # in_ch = args.input_channels
        time_emb_scale=1.0
        num_groups=16
        res_scale=0.2
        self.res_scale = res_scale
        #rate = [1,1,1,2,2,4,4]
        # in_ch = 1
        out_ch = 1
        ch = [base_channels] + [base_channels * i for i in rate]
        use_attention = args.use_attention
        down_mamba = args.down_mamba
        up_mamba = args.up_mamba
        bottleneck_mamba = args.bottleneck_mamba
        assert len(use_attention) == len(rate) == len(down_mamba) == len(up_mamba), f'{use_attention}--{rate}--{down_mamba}--{up_mamba}'

        self.in_block_x1 = DoubleConv(args.x_channels, base_channels, mid_channels=int(base_channels//2) if int(base_channels//2) > args.x_channels else None)
        self.in_block_x2 = DoubleConv(args.y_channels, base_channels, mid_channels=int(base_channels//2) if int(base_channels//2) > args.y_channels else None)
        self.down_blocks_x1 = nn.ModuleList()
        self.down_blocks_x2 = nn.ModuleList()
        print(f'double conv: in-{args.x_channels}/{args.y_channels},out-{base_channels}')

        self.attention_blocks = nn.ModuleList()
        for i in range(len(ch)-1):
            ng = num_groups
            while (ng%ch[i+1] != 0 or ng%ch[i+1] != 0) and ng != 1: ng//=2
            print(f'down: in-{ch[i]},out-{ch[i+1]}')
            self.down_blocks_x1.append(
                Down_block_PVM(ch[i], ch[i+1], base_channels, time_emb_scale, num_groups=ng)
                if down_mamba[i] else
                Down_block(ch[i], ch[i+1], base_channels, time_emb_scale, num_groups=ng)
                                    ) 
            self.down_blocks_x2.append(
                Down_block_PVM(ch[i], ch[i+1], base_channels, time_emb_scale, num_groups=ng)
                if down_mamba[i] else
                Down_block(ch[i], ch[i+1], base_channels, time_emb_scale, num_groups=ng)
                                    )
            self.attention_blocks.append(self_attention2(ch[i+1], reduce=32) if use_attention[i]==True else empty())
        # self.attention_blocks[-1] = nn.Sequential()

        ng = num_groups
        while ng%ch[-1] != 0 and ng != 1:
                ng//=2
        print('bottlenek, ch=', ch[-1])

        self.bottleneck = Bottleneck_mamba(ch[-1], ng) if bottleneck_mamba else Bottleneck(ch[-1], ng)
        
        self.up_blocks = nn.ModuleList()

        self.Bridge = {
            1: SC_Att_Bridge,
            2: SC_Att_Bridge2,
            3: SC_Att_Bridge2
        }[args.bridge](ch[1:])

        for i in reversed(range(len(ch)-1)):
            ng = num_groups
            while (ng%ch[i+1] != 0 or ng%ch[i+1] != 0) and ng != 1:
                ng//=2
            print(f'up block: in-{ch[i+1]},out-{ch[i]}')
            self.up_blocks.append(
                Up_block_mamba(ch[i+1], ch[i], base_channels, time_emb_scale,num_groups=ng)
                if up_mamba[i] else
                Up_block(ch[i+1], ch[i], base_channels, time_emb_scale,num_groups=ng)
                )

        self.out_block = nn.Sequential(
            nn.BatchNorm2d(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_ch, 3, padding=1)
        )

    
    def forward(self, x1, t, y):  # x1 for xt, x2 for image
        x1 = self.in_block_x1(x1)
        x2 = self.in_block_x2(y)
        out_x1 = []
        out_x2 = []
        d_x1_, d_x1 = None, x1
        d_x2_, d_x2 = None, x2
        for down_block_x1, down_block_x2, attention_block in zip(self.down_blocks_x1, self.down_blocks_x2, self.attention_blocks):
            d_x1_, d_x1 = down_block_x1(d_x1, t)
            d_x2_, d_x2 = down_block_x2(d_x2, t)
            d_x1, d_x2 = attention_block(d_x1, d_x2)
            out_x1.append(d_x1_)
            out_x2.append(d_x2_)

        u = self.bottleneck(d_x1 + d_x2)
        out_x = self.Bridge(out_x1, out_x2)


        # u = m + self.down3(resnet_map).mul(self.res_scale)
        # u = m + resnet_map.mul(self.res_scale)
        for up_block, d_x in zip(self.up_blocks, reversed(out_x)):
            # print('*'*20)
            u = up_block(u, d_x, t)
        return self.out_block(u)


@Registers.runners.register_with_name('train01')
class Unet7(Unet5):  # from Unet5
    def __init__(self, args):
        super().__init__(args)
    
    def forward(self, y):  # x1 for xt, x2 for image
        x1 = self.in_block_x1(y)
        x2 = self.in_block_x2(y)
        out_x1 = []
        out_x2 = []
        d_x1_, d_x1 = None, x1
        d_x2_, d_x2 = None, x2
        for down_block_x1, down_block_x2, attention_block in zip(self.down_blocks_x1, self.down_blocks_x2, self.attention_blocks):
            d_x1_, d_x1 = down_block_x1(d_x1, t=None)
            d_x2_, d_x2 = down_block_x2(d_x2, t=None)
            d_x1, d_x2 = attention_block(d_x1, d_x2)
            out_x1.append(d_x1_)
            out_x2.append(d_x2_)
        
        u = self.bottleneck(self.merge_before_bn(d_x1, d_x2))


        for up_block, connection, d_x1_, d_x2_ in zip(self.up_blocks, self.connections, reversed(out_x1), reversed(out_x2)):
            u = up_block(u, connection(d_x1_, d_x2_), t=None)
        return self.out_block(u)