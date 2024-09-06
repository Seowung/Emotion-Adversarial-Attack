import sys
import torch
import torch.nn as nn
from torch.autograd import Function, Variable

class ClassWisePoolFunction(Function):
    @staticmethod
    def forward(ctx, input, num_maps):
        ctx.num_maps = num_maps
        # batch dimension
        batch_size, num_channels, h, w = input.size()

        if num_channels % ctx.num_maps != 0:
            print('Error in ClassWisePoolFunction. The number of channels has to be a multiple of the number of maps per class')
            sys.exit(-1)

        num_outputs = int(num_channels / ctx.num_maps)
        x = input.view(batch_size, num_outputs, ctx.num_maps, h, w)
        output = torch.sum(x, 2)
        ctx.save_for_backward(input)
        return output.view(batch_size, num_outputs, h, w) / ctx.num_maps

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # batch dimension
        batch_size, num_channels, h, w = input.size()
        num_outputs = grad_output.size(1)

        grad_input = grad_output.view(batch_size, num_outputs, 1, h, w).expand(batch_size, num_outputs, ctx.num_maps,
                                                                               h, w).contiguous()
        return grad_input.view(batch_size, num_channels, h, w), None

class ClassWisePool(nn.Module):
    def __init__(self, num_maps):
        super(ClassWisePool, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        return ClassWisePoolFunction.apply(input, self.num_maps)