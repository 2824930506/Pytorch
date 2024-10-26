import torch
from propagation import *
device = torch.device('cuda')
# 创建抛物相位透镜
def create_lens_phase(size, wavelength, f, pixel_size):
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, size),
        torch.linspace(-1, 1, size)
    )
    r2 = (x**2 + y**2) * (size / 2 * pixel_size) ** 2
    phase = torch.exp(-1j * torch.pi * r2 / (wavelength * f))
    return phase.to(device)
# 向前传播
def single_shot_ptychograph(obj, pixel_size, wavelength,z, f1, f2, mask):
    """
    obj:
    :return:
    """
    # LEN 1
    size = obj.size()
    fields = propagation_ASM(obj, pixel_size, wavelength, z[0]) # 传播到第一个透镜前表面
    Len_1 = create_lens_phase(size[3], wavelength, f1, pixel_size[0])
    fields = fields*Len_1 # 与透镜传递函数相乘
    fields = propagation_ASM(fields, pixel_size, wavelength, z[1])
    fields = fields*mask
    # LEN 2
    fields = propagation_ASM(fields, pixel_size, wavelength, z[2])
    Len_2 = create_lens_phase(size[3], wavelength, f2, pixel_size[1])
    fields = fields * Len_2  # 与透镜传递函数相乘
    fields = propagation_ASM(fields, pixel_size, wavelength, z[3])
    return fields


