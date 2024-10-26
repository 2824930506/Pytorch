import cv2
import torch
from propagation import *
# 定义调整图片大小的函数
def preprocess_image(img_path):
    # 读取图片（默认是BGR格式）
    img = cv2.imread(img_path)

    # 将图片转换为灰度图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 调整图片大小为 32x32
    resized_img = cv2.resize(gray_img, (32, 32))

    # 将图片转换为 [1, H, W] 格式，并归一化到 [0, 1] 区间
    tensor_img = torch.tensor(resized_img, dtype=torch.float32).unsqueeze(0) / 255.0

    return tensor_img
# 读取并处理三张图片
img1_tensor = preprocess_image('C:/Users/Administrator/Downloads/cifar-10-batches-mat/horse/test_image_1_label_0.jpg')
print('\n',img1_tensor.shape)
img2_tensor = preprocess_image('C:/Users/Administrator/Downloads/cifar-10-batches-mat/horse/test_image_2_label_6.jpg')
img3_tensor = preprocess_image('C:/Users/Administrator/Downloads/cifar-10-batches-mat/horse/test_image_3_label_0.jpg')

# 堆叠图片形成 [3, 1, 32, 32] 的张量
image_tensor = torch.stack([img1_tensor, img2_tensor, img3_tensor])  # [3, 1, 32, 32]

# 将张量转换为 [1, 3, 32, 32]，其中第一个维度为批次维度
image_tensor = image_tensor.permute(1, 0, 2, 3)  # [1, 3, 32, 32]

print(image_tensor.shape)  # 输出 (1, 3, 32, 32)


pattern = propagation_ASM(image_tensor, [3.45e-6, 3.45e-6], 520e-9, 0.01)
