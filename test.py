import torch

class _TruncateActivationRange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a_bit):
        ctx.a_bit = a_bit
        binary_mask = (- 2 ** (a_bit - 1) <= x) & (x <= 2 ** (a_bit - 1) - 1)
        ctx.save_for_backward(binary_mask)
        return x.clamp(- 2 ** (a_bit - 1), 2 ** (a_bit - 1) - 1)

    @staticmethod
    def backward(ctx, grad_output):
        binary_mask, = ctx.saved_tensors
        grad_x = grad_output * binary_mask
        return grad_x, None

out = -78 * torch.ones(3,3,dtype=torch.int8)
print(out)
y=_TruncateActivationRange.apply(out, 6)
print(y)