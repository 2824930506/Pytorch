from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. 加载图片
image_path = "C:/Users/Administrator/Downloads/cifar-10-batches-mat/horse/test_image_958_label_8.jpg"
image = Image.open(image_path)

# 2. 将图片转为灰度图像
gray_image = image.convert("L")

# 3. 将灰度图像转为 NumPy 数组，用于轮廓检测
gray_np = np.array(gray_image)

# 4. 使用 OpenCV 进行二值化
_, binary = cv2.threshold(gray_np, 127, 255, cv2.THRESH_BINARY_INV)

# 5. 使用 OpenCV 找出轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 6. 创建一个与图像大小一致的黑色掩膜
mask = np.zeros_like(gray_np)
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

# 7. 使用掩膜从原图抠出对象
masked_image = np.array(image) * np.repeat(mask[:, :, np.newaxis], 3, axis=2) // 255

# 8. 显示结果，禁用插值
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(np.array(image))
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(gray_image, cmap='gray', interpolation='nearest')
ax[1].set_title("Grayscale Image")
ax[1].axis("off")

ax[2].imshow(masked_image, interpolation='nearest')
ax[2].set_title("Extracted Horse")
ax[2].axis("off")

plt.show()
