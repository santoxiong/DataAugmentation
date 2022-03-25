import matplotlib.pyplot as plt
import numpy as np
import PIL
import os
import math
from PIL import Image
from F import DataAugmentation as FD
import imageio
import imgaug


def HorizontalShift(image):
    random_xoffset = np.random.randint(0, math.ceil(image.size[0] * 0.2))
    return image.offset(random_xoffset)


def VerticalShift(image):
    random_yoffset = np.random.randint(0, math.ceil(image.size[1] * 0.2))
    return image.offset(random_yoffset)


# 极性翻转 PF
def PolarityFlip(img_path, save_path):
    image = Image.open(img_path)
    X = np.array(image)
    S = X * -1

    width, height = S.shape
    plt.figure(figsize=(width / 100, height / 100))
    plt.imshow(S, cmap='gray')
    plt.axis('off')
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)


# 水平翻转 HF
def HorizontalFlip(img_path, save_path):
    image = FD.openImage(img_path)
    new_image = FD.randomFlip(image, mode=Image.FLIP_LEFT_RIGHT)
    FD.saveImage(new_image, save_path)


# 随意剪切 RC
def RandomCrop(img_path, save_path):
    image = FD.openImage(img_path)
    new_image = FD.randomCrop(image)
    FD.saveImage(new_image, save_path)


# # 水平平移 HT
# def HorizontalTranslating(img_path, save_path):
#     image = FD.openImage(img_path)
#     new_image = HorizontalShift(image)
#     FD.saveImage(new_image, save_path)
#
#
# # 垂直平移 VT
# def VerticalTranslating(img_path, save_path):
#     image = FD.openImage(img_path)
#     new_image = VerticalShift(image)
#     FD.saveImage(new_image, save_path)


# 水平拉伸 HS
def HorizontalStretch(img_path, save_path):
    image = FD.openImage(img_path)
    w, h = image.size
    new_image = image.resize((int(1.5 * w), int(h)))
    FD.saveImage(new_image, save_path)


# 水平压缩 HC
def HorizontalCompression(img_path, save_path):
    image = FD.openImage(img_path)
    w, h = image.size
    new_image = image.resize((int(0.5 * w), int(h)))
    FD.saveImage(new_image, save_path)


# 垂直拉伸 VS
def VerticalStretch(img_path, save_path):
    image = FD.openImage(img_path)
    w, h = image.size
    new_image = image.resize((int(w), int(1.5 * h)))
    FD.saveImage(new_image, save_path)


# 垂直压缩 VC
def VerticalCompression(img_path, save_path):
    image = FD.openImage(img_path)
    w, h = image.size
    new_image = image.resize((int(w), int(0.5 * h)))
    FD.saveImage(new_image, save_path)


# 高斯白噪声 GN
def GaussianNoisy(img_path, save_path):
    image = FD.openImage(img_path)
    # new_image = FD.gaussianNoisy(image, mean=0, sigma=10)
    new_image = FD.randomGaussian(image, mean=100, sigma=200)
    FD.saveImage(new_image, save_path)


def awgn(img_path, save_path):
    np.random.seed(8)
    snr = 50
    image = FD.openImage(img_path)
    w, h = image.size
    for i in range(w):
        x = image[w, :]
        SNR = 10 ** (snr / 10.0)
        x_power = np.sum(x ** 2) / w
        n_power = x_power / SNR
        noise = np.random.randn(w) * np.sqrt(n_power)
        new_x = x + noise


if __name__ == '__main__':
    # img_dir = os.path.join(absolute_path, 'Images/')
    # images = os.listdir(img_dir)
    # for image in images:
    #     name = image.split('.')[0]
    #     img_path = os.path.join(img_dir, image)
    #     save_path = os.path.join(absolute_path, f'')
    absolute_path = os.getcwd()
    img_dir = os.path.join(absolute_path, 'Images/6-15-3-1_2.jpg')
    PolarityFlip(img_path=img_dir, save_path=os.path.join(absolute_path, 'PF/6-15-3-1_2PF.jpg'))
    HorizontalFlip(img_path=img_dir, save_path=os.path.join(absolute_path, 'HF/6-15-3-1_2HF.jpg'))
    RandomCrop(img_path=img_dir, save_path=os.path.join(absolute_path, 'RC/6-15-3-1_2RC.jpg'))
    HorizontalCompression(img_path=img_dir, save_path=os.path.join(absolute_path, 'HC/6-15-3-1_2HC.jpg'))
    VerticalCompression(img_path=img_dir, save_path=os.path.join(absolute_path, 'VC/6-15-3-1_2VC.jpg'))
    HorizontalStretch(img_path=img_dir, save_path=os.path.join(absolute_path, 'HS/6-15-3-1_2HS.jpg'))
    VerticalStretch(img_path=img_dir, save_path=os.path.join(absolute_path, 'VS/6-15-3-1_2VS.jpg'))
