# YOLOv5 ğŸš€ by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

device = torch.device("cuda:0")
class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy.to(device) * 2 + self.grid[i].to(device)) * self.stride[i].to(device)  # xy
                    wh = (wh.to(device) * 2) ** 2 * self.anchor_grid[i].to(device)  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    # YOLOv5 base model
    def forward(self, x, x_A, profile=False, visualize=False):
        # x = torch.cat([x,x_A],1)
        # x_s = x - x_A
        # x_d = x + x_A
        # x = torch.cat([x,x_A,x_s,x_d],1)
        return self._forward_once(x,x_A, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, x_A, profile=False, visualize=False):
        y, y_A, y_H, dt = [], [], [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        for m_A in self.model_A:
            if profile:
                self._profile_one_layer(m_A, x_A, dt)
            x_A = m_A(x_A)
            y_A.append(x_A if m_A.i in self.save else None)

        x_H = torch.cat([x, x_A], 1)
        for m_H in self.model_H:
            if m_H.f != -1 and m_H.f != -2:  # if not from previous layer
                if isinstance(m_H.f, int):
                    x_H = y[m_H.f] - y_A[m_H.f]
                else:  
                    x_H = []
                    for j in m_H.f:
                        if j == -1:
                           x_H.append(y_H[-1]) 
                        if j <10 and j != -1:
                            x_H.append( y[j] - y_A[j] )
                        if j >10:
                            x_H.append(y_H[j-11]) 
            if m_H.f == -2:
                x_H = y_H[-2]
            if profile:
                self._profile_one_layer(m_H, x_H, dt)
            
            x_H = m_H(x_H)
            y_H.append(x_H)
        
        return x_H

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model_H[-1]   # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv5 detection model 加载配置文件
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict 加载模型 self.yaml存放的是yolov5s.yaml中关键字和值的格式，以python内置字典类型存放

        # Define model 搭建网络每一层
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels 从字典中取出'ch'关键字所表示的值，如果取不到则用('ch', ch)中ch的值，这里相当于往yaml中追加了一个'ch'关键字
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value 如果传入的nc和配置文件里的不等则用新值覆盖
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.model_A, self.model_H, self.save  = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist 利用yaml文件一步步地搭建网络每一层 ch=[ch*2]
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names 给每一类赋一个类名0，1，2...
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors 
        m = self.model_H[-1] # Detect() -1表示取出模块的最后一层
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x, x_A: self.forward(x, x_A)[0] if isinstance(m, Segment) else self.forward(x, x_A)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s), torch.zeros(1, ch, s, s))])  # forward 新建一个256*256空图片，通过一次前向传播得到步长列表[8,16,32]
            check_anchor_order(m) #判断anchors有没有写反，若顺序不对则调整顺序
            m.anchors /= m.stride.view(-1, 1, 1) #将相对于原图的anchor按照步长缩放为相对于不同尺度特征图大小的anchor
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases 初始化及打印
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, x_A, augment=False, profile=False, visualize=False):
        # x = torch.cat([x,x_A],1)
        # x_s = x - x_A
        # x_d = x + x_A
        # x = torch.cat([x,x_A,x_s,x_d],1)
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, x_A, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model_H[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model_H[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None


def parse_model(d, ch):  # d:model_dict, ch:input_channels(3)
    # Parse a YOLOv5 model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}") #打印信息
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation') #取出值并赋值
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors anchors中有3个列表，每个中有6个元素，na=3
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5) 模型最终输出的通道数  3*(80+5)=255
    
    # 第一个backbone支路
    layers, save, c2 = [], [], ch[-1]  # layers存储下面搭建的每一层, savelist：save保存某层是否被保存并备后续使用 例如C3 C6等， c2：ch out输出通道，c1表示输入
    ch_A = ch    
    for i, (f, n, m, args) in enumerate(d['backbone'] ):  # 以第0层为例 from:-1, number:1, module:'Conv', args:[64,6,2,2]
        m = eval(m) if isinstance(m, str) else m  # eval strings 利用eval进行推断，得出m:<class 'models.commn.Conv'>
        for j, a in enumerate(args): #遍历args中的每一个元素
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings  [64, 6, 2, 2]

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain 求出真正的层数，例如C3模块3*0.33 再round四舍五入
        if m in {
                Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}: # 判断m属于什么结构
            c1, c2 = ch[f], args[0] #c1:3 c2:64
            if c2 != no:  # if not output 除了输出层中间各层都需要乘上通道倍数，即64*0.5=32，再判断是不是8的倍数，若不是则变成8的倍数
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]] #拼接起来形成新的args[3,32,6,2,2] 以符合commn中卷积层初始化需要
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}: #这几个模块单独再处理一下，
                args.insert(2, n)  # number of repeats 将n拼接到args的第2位
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
            
        #nn.Sequential(),一个序列容器，用于搭建神经网络的模块被按照被传入构造器的顺序添加到nn.Sequential()容器中。
        #利用nn.Sequential()搭建好模型架构，模型前向传播时调用forward()方法，模型接收的输入首先被传入nn.Sequential()包含的第一个网络模块中。
        #然后，第一个网络模块的输出传入第二个网络模块作为输入，按照顺序依次计算并传播，直到nn.Sequential()里的最后一个模块输出结果。
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module 根据n是否大于1判断初始化多少个模块  
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params 统计参数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params 将值赋值给某层属性
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print 打印输出信息
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1 )  # append to savelist 储存要额外保存的层 [6,4,14,10,17,20,23] -> [4,6,10,14,17,20,23]
        layers.append(m_)
        if i == 0:
            ch = [] 
        ch.append(c2) # [32], [32,64], [32,64,64]... 添加输出通道到ch列表，取出上一层输出通道作为本层输入通道
    
    # # 第二个backbone支路 
    layers_A, save_A, c2_A = [], [], ch_A[-1]  # layers存储下面搭建的每一层, savelist：save保存某层是否被保存并备后续使用 例如C3 C6等， c2：ch out输出通道，c1表示输入
    for i_A, (f_A, n_A, m_A, args_A) in enumerate(d['backbone_A'] ):  # 以第0层为例 from:-1, number:1, module:'Conv', args:[64,6,2,2]
        m_A = eval(m_A) if isinstance(m_A, str) else m_A  # eval strings 利用eval进行推断，得出m:<class 'models.commn.Conv'>
        for j_A, a_A in enumerate(args_A): #遍历args中的每一个元素
            with contextlib.suppress(NameError):
                args_A[j_A] = eval(a_A) if isinstance(a_A, str) else a_A  # eval strings  [64, 6, 2, 2]

        n_A = n__A = max(round(n_A * gd), 1) if n_A > 1 else n_A  # depth gain 求出真正的层数，例如C3模块3*0.33 再round四舍五入
        if m_A in {
                Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}: # 判断m属于什么结构
            c1_A, c2_A = ch_A[f_A], args_A[0] #c1:3 c2:64
            if c2_A != no:  # if not output 除了输出层中间各层都需要乘上通道倍数，即64*0.5=32，再判断是不是8的倍数，若不是则变成8的倍数
                c2_A = make_divisible(c2_A * gw, 8)

            args_A = [c1_A, c2_A, *args_A[1:]] #拼接起来形成新的args[3,32,6,2,2] 以符合commn中卷积层初始化需要
            if m_A in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}: #这几个模块单独再处理一下，
                args_A.insert(2, n_A)  # number of repeats 将n拼接到args的第2位
                n_A = 1
        elif m_A is nn.BatchNorm2d:
            args_A = [ch_A[f_A]]
        elif m_A is Concat:
            c2_A = sum(ch_A[x] for x in f_A)
        # TODO: channel, gw, gd
        elif m_A in {Detect, Segment}:
            args_A.append([ch_A[x] for x in f_A])
            if isinstance(args_A[1], int):  # number of anchors
                args_A[1] = [list(range(args_A[1] * 2))] * len(f_A)
            if m_A is Segment:
                args_A[3] = make_divisible(args_A[3] * gw, 8)
        elif m_A is Contract:
            c2_A = ch_A[f_A] * args_A[0] ** 2
        elif m_A is Expand:
            c2_A = ch_A[f_A] // args_A[0] ** 2
        else:
            c2_A = ch_A[f_A]

        m__A = nn.Sequential(*(m_A(*args_A) for _ in range(n_A))) if n_A > 1 else m_A(*args_A)  # module 根据n是否大于1判断初始化多少个模块  
        t_A = str(m_A)[8:-2].replace('__main__.', '')  # module type
        np_A = sum(x.numel() for x in m__A.parameters())  # number params 统计参数量
        m__A.i, m__A.f, m__A.type, m__A.np = i_A, f_A, t_A, np_A  # attach index, 'from' index, type, number params 将值赋值给某层属性
        LOGGER.info(f'{i_A:>3}{str(f_A):>18}{n__A:>3}{np_A:10.0f}  {t_A:<40}{str(args_A):<30}')  # print 打印输出信息
        save_A.extend(x % i_A for x in ([f_A] if isinstance(f_A, int) else f_A) if x != -1)  # append to savelist 储存要额外保存的层
        layers_A.append(m__A)
        if i_A == 0:
            ch_A = [] 
        ch_A.append(c2_A) # [32], [32,64], [32,64,64]... 添加输出通道到ch列表，取出上一层输出通道作为本层输入通道

    # # 目标检测head路径 
    ch_H =  ch
    ch_H[-1] = ch_H[-1]*2
    layers_H, save_H, c2_H = [], [], ch_H[-1]  # layers存储下面搭建的每一层, savelist：save保存某层是否被保存并备后续使用 例如C3 C6等， c2：ch out输出通道，c1表示输入
    for i_H, (f_H, n_H, m_H, args_H) in enumerate(d['head'] ):  # 以第0层为例 from:-1, number:1, module:'Conv', args:[64,6,2,2]     
        i_H += 11
        m_H = eval(m_H) if isinstance(m_H, str) else m_H  # eval strings 利用eval进行推断，得出m:<class 'models.commn.Conv'>
        for j_H, a_H in enumerate(args_H): #遍历args中的每一个元素
            with contextlib.suppress(NameError):
                args_H[j_H] = eval(a_H) if isinstance(a_H, str) else a_H  # eval strings  [64, 6, 2, 2]

        n_H = n__H = max(round(n_H * gd), 1) if n_H > 1 else n_H  # depth gain 求出真正的层数，例如C3模块3*0.33 再round四舍五入
        if m_H in {
                Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}: # 判断m属于什么结构
            c1_H, c2_H = ch_H[f_H] , args_H[0] #c1:3 c2:64
            if c2_H != no:  # if not output 除了输出层中间各层都需要乘上通道倍数，即64*0.5=32，再判断是不是8的倍数，若不是则变成8的倍数
                c2_H = make_divisible(c2_H * gw, 8)

            args_H = [c1_H, c2_H, *args_H[1:]] #拼接起来形成新的args[3,32,6,2,2] 以符合commn中卷积层初始化需要
            if m_H in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}: #这几个模块单独再处理一下，
                args_H.insert(2, n_H)  # number of repeats 将n拼接到args的第2位
                n_H = 1
        elif m_H is nn.BatchNorm2d:
            args_H = [ch_H[f_H]]
        elif m_H is Concat:
            c2_H = sum(ch_H[x] for x in f_H)
        # TODO: channel, gw, gd
        elif m_H in {Detect, Segment}:
            args_H.append([ch_H[x] for x in f_H])
            if isinstance(args_H[1], int):  # number of anchors
                args_H[1] = [list(range(args_H[1] * 2))] * len(f_H)
            if m_H is Segment:
                args_H[3] = make_divisible(args_H[3] * gw, 8)
        elif m_H is Contract:
            c2_H = ch_H[f_H] * args_H[0] ** 2
        elif m_H is Expand:
            c2_H = ch_H[f_H] // args_H[0] ** 2
        else:
            c2_H = ch_H[f_H]

        m__H = nn.Sequential(*(m_H(*args_H) for _ in range(n_H))) if n_H > 1 else m_H(*args_H)  # module 根据n是否大于1判断初始化多少个模块  
        t_H = str(m_H)[8:-2].replace('__main__.', '')  # module type
        np_H = sum(x.numel() for x in m__H.parameters())  # number params 统计参数量
        m__H.i, m__H.f, m__H.type, m__H.np = i_H, f_H, t_H, np_H  # attach index, 'from' index, type, number params 将值赋值给某层属性
        LOGGER.info(f'{i_H:>3}{str(f_H):>18}{n__H:>3}{np_H:10.0f}  {t_H:<40}{str(args_H):<30}')  # print 打印输出信息
        save_H.extend(x % i_H for x in ([f_H] if isinstance(f_H, int) else f_H) if x != -1)  # append to savelist 储存要额外保存的层
        layers_H.append(m__H)
        if i_H == 0:
            ch_H = [] 
        ch_H.append(c2_H) # [32], [32,64], [32,64,64]... 添加输出通道到ch列表，取出上一层输出通道作为本层输入通道

    return nn.Sequential(*layers), nn.Sequential(*layers_A), nn.Sequential(*layers_H), sorted(save_H)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov3.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 512, 512).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()

#统计参数量及计算效率
    from thop import profile
    img = torch.rand(1, 3, 512, 512).to(device)
    flops, params = profile(model, inputs=(img,img ))
    print("params: %.2fMB ------- flops: %.2fG" % (params / (1000 ** 2), flops / (1000 ** 3)))
    print('flops:', flops)
    print('params:', params)