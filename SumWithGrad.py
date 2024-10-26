import torch
class SumWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        # 在 forward 中执行求和操作
        ctx.save_for_backward(input_tensor)
        return (torch.sum(input_tensor, dim=1)).unsqueeze(dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        # 获取正向传播时保存的张量
        input_tensor, = ctx.saved_tensors
        # 计算梯度：每个元素的梯度都是 1，所以直接乘以 grad_output
        grad_input = grad_output * torch.ones_like(input_tensor)
        return grad_input