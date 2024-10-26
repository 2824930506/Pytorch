from complex_generator import ComplexGenerator
from holo_encoder import HoloEncoder
from utils import *
from propagation import *
import torch.nn.functional as F
from SSP import *
device = torch.device('cuda')
class selfholo(nn.Module):

    def __init__(self):
        super().__init__()
        self.network1 = ComplexGenerator() # 3D fields
        self.network2 = HoloEncoder()      # diffraction pattern

        self.wavelength = 520e-9
        self.feature_size = [3.45e-6, 3.45e-6]   # pixel_size=3.45微米 dx,dy
        self.z = [0.05, 0.047, 0.057, 0.05]                       #
        self.precomputed_H = None
        self.pren = None
        self.prem = None
        self.pref = None
        self.return_H = None
        self.f_mla = 100e-3
        self.N_mla = 64  # 图像大小 (像素)
        self.f1 = 100e-3
        self.f2 = 100e-3

    def forward(self, source, ikk):
        source_size = source.size()
        """
        print(source.shape)
        if self.precomputed_H == None:
            self.precomputed_H = propagation_ASM(torch.empty(1, 1, 1072, 1072), feature_size=[8e-6, 8e-6],
                                                 wavelength=self.wavelength, z=0.30, return_H=True)
            self.precomputed_H = self.precomputed_H.to('cuda').detach()
            self.precomputed_H.requires_grad = False

        if self.pren == None:
            self.pren = propagation_ASM(torch.empty(1, 1, 1072, 1072), feature_size=[8e-6, 8e-6],
                                        wavelength=self.wavelength, z=-0.30, return_H=True)
            self.pren = self.pren.to('cuda').detach()
            self.pren.requires_grad = False

        if self.prem == None:
            self.prem = propagation_ASM(torch.empty(1, 1, 1072, 1072), feature_size=[8e-6, 8e-6],
                                        wavelength=self.wavelength, z=-0.31, return_H=True)
            self.prem = self.prem.to('cuda').detach()
            self.prem.requires_grad = False

        if self.pref == None:
            self.pref = propagation_ASM(torch.empty(1, 1, 1072, 1072), feature_size=[8e-6, 8e-6],
                                        wavelength=self.wavelength, z=-0.32, return_H=True)
            self.pref = self.pref.to('cuda').detach()
            self.pref.requires_grad = False
        """
        # 前向传播
        fields = self.network2(source)       # 只用 network2 就可以 这里输出应该是3张图
        # 相位因子设置
        # 生成网格坐标
        x = torch.linspace(-self.N_mla / 2, self.N_mla / 2 - 1, self.N_mla)
        y = torch.linspace(-self.N_mla / 2, self.N_mla / 2 - 1, self.N_mla)
        x, y = torch.meshgrid(x, y)
        r = torch.sqrt(x ** 2 + y ** 2) * self.feature_size[0]  # 计算每个像素到中心的半径
        # 计算相位因子
        phase = torch.exp(-1j * np.pi * r ** 2 / (self.wavelength * self.f_mla))  # 正常相位分布
        radius_threshold = 100e-6  # 设置圆的半径阈值 (m)
        phase[r > radius_threshold] = 0  # 将半径大于阈值的区域设为0
        phase = torch.tile(phase, (3, 3))
        phase = torch.reshape(phase, (1, 1, *phase.size()))
        fields_size = phase.size()
        phase = phase.repeat(1, fields_size[1], 1, 1)
        pad_height = (source_size[2] - fields_size[2]) // 2  # Padding on top and bottom
        pad_width = (source_size[3] - fields_size[3]) // 2  # Padding on left and right
        phase = F.pad(phase, (pad_width, pad_width, pad_height, pad_height))
        phase = phase.to(device)
        cc = fields*phase
        # (obj, pixel_size, wavelength, z, f1, f2, mask)
        slm_field = single_shot_ptychograph(phase, self.feature_size, self.wavelength, self.z, self.f1, self.f2, fields)    #这里输入应该是3张图
        print('slm_field.shape:', slm_field.shape)
        diff_pattern = (torch.sum(slm_field, dim=1)).unsqueeze(dim=1)                         # 解决sum的问题
        diff_pattern = diff_pattern.real**2+diff_pattern.imag**2

        return diff_pattern