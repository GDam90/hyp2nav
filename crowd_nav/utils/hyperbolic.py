# from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from torch import nn
import torch
import math
import torch.nn.init as init
from crowd_nav.utils import pmath
import geoopt.manifolds.stereographic.math as gmath

class HypLinear(nn.Module):
    def __init__(self, in_features, out_features, c=-1, bias=True):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, c=torch.tensor(-1.0)):
        if c is None:
            c = self.c
        mv = gmath.mobius_matvec(self.weight, x, k=c)
        if self.bias is None:
            return gmath.project(mv, k=c)
        else:
            bias = gmath.expmap0(self.bias, k=c)
            return gmath.project(gmath.mobius_add(mv, bias, k=c), k=c)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, c={}".format(
            self.in_features, self.out_features, self.bias is not None, self.c
        )
    
class ExpMapProjectionLayer(nn.Module):
    def __init__(self, c=-1):
        super(ExpMapProjectionLayer, self).__init__()
        self.c = c

    def forward(self, x, c=None):
        k = self.c if c is None else c
        return gmath.project(gmath.expmap0(x, k=k), k=k)