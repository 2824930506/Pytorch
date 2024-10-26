import torch
import torch.fft as fft

# 初始化参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 256  # 图像尺寸（NxN）

# 随机生成复数形式的探针和物体
def generate_complex_image(size):
    real = torch.rand(size, size, device=device)
    imag = torch.rand(size, size, device=device)
    return real + 1j * imag  # 生成复数图像

probe = generate_complex_image(img_size)  # 探针
obj = generate_complex_image(img_size)    # 物体

# 前向传播模型：计算出射波的傅里叶变换（衍射图样）
def forward_model(probe, obj):
    exit_wave = probe * obj  # 探针和物体的逐元素相乘
    diffraction_pattern = fft.fftn(exit_wave)  # 进行2D傅里叶变换
    intensity_pattern = torch.abs(diffraction_pattern) ** 2  # 计算强度图
    return intensity_pattern

# 执行前向传播，生成衍射图样
intensity = forward_model(probe, obj)

# 打印结果形状，确保尺寸正确
print(f"衍射图样尺寸: {intensity.shape}")
