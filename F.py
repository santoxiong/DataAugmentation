import logging
import math
import os
import random
import shutil

import numpy as np
from PIL import Image, ImageEnhance, ImageFile
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataAugmentation:
    """
    数据增强8种方式：
       1. 翻转变换 flip
       2. 随机修剪 random crop
       3. 色彩抖动 color jittering
       4. 平移变换 shift
       5. 尺度变换 scale
       6. 对比度变换 contrast
       7. 噪声扰动 noise
       8. 旋转变换/反射变换 Rotation/reflection
    """

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")

    @staticmethod
    def randomFlip(image, mode=Image.FLIP_LEFT_RIGHT):
        """
        对图像进行上下左右四个方面的随机翻转
        :param image: PIL的图像image
        :param model: 水平或者垂直方向的随机翻转模式,默认右向翻转
        :return: 翻转之后的图像

        model
    - Image.FLIP_LEFT_RIGHT,表示将图像左右翻转
    - Image.FLIP_TOP_BOTTOM,表示将图像上下翻转
    - Image.ROTATE_90,表示将图像逆时针旋转90°
    - Image.ROTATE_180,表示将图像逆时针旋转180°
    - Image.ROTATE_270,表示将图像逆时针旋转270°
    - Image.TRANSPOSE,表示将图像进行转置(相当于顺时针旋转90°)
    - Image.TRANSVERSE,表示将图像进行转置,再水平翻转
        """
        # random_model = np.random.randint(0, 2)
        # flip_model = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
        # return image.transpose(flip_model[random_model])
        return image.transpose(mode)

    @staticmethod
    def randomShift(image):
        # def randomShift(image, xoffset, yoffset=None):
        """
        对图像进行平移操作
        :param image: PIL image
        :param xoffset: x方向向右平移
        :param yoffset: y方向向下平移
        :return: 翻转之后的图像
        """
        random_xoffset = np.random.randint(0, math.ceil(image.size[0] * 0.2))
        random_yoffset = np.random.randint(0, math.ceil(image.size[1] * 0.2))
        # return image.offset(xoffset = random_xoffset, yoffset = random_yoffset)
        return image.offset(random_xoffset)

    # @staticmethod
    # def randomRotation(image, mode=Image.BICUBIC):
    #     """
    #      对图像进行随机任意角度(0~360度)旋转
    #     :param mode 邻近插值,双线性插值,双三次B样条插值(default)
    #     --PIL.Image.BICUBIC
    #     --PIL.Image.LANCZOS
    #     --PIL.Image.BILINEAR
    #     --PIL.Image.NEAREST
    #
    #     :param image PIL的图像image
    #     :return: 旋转转之后的图像
    #     """
    #     random_angle = np.random.randint(1, 360)
    #     return image.rotate(random_angle, mode)

    @staticmethod
    def randomCrop(image):
        """
        对图像随意剪切,裁剪图像大小宽和高的2/3
        :param image: PIL的图像image
        :return: 剪切之后的图像
        """
        image_width = image.size[0]
        image_height = image.size[1]
        crop_image_width = math.ceil(image_width * 2 / 3)
        crop_image_height = math.ceil(image_height * 2 / 3)
        x = np.random.randint(0, image_width - crop_image_width)
        y = np.random.randint(0, image_height - crop_image_height)
        random_region = (x, y, x + crop_image_width, y + crop_image_height)
        return image.crop(random_region)

    # @staticmethod
    # def randomColor(image):
    #     """
    #     对图像进行颜色抖动
    #     :param image: PIL的图像image
    #     :return: 有颜色色差的图像image
    #     """
    #     random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    #     color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    #     random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    #     brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    #     random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
    #     contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    #     random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    #     return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

    @staticmethod
    def randomGaussian(image, mean=0.2, sigma=0.3):
        """
         对图像进行高斯噪声处理
        :param image:
        :return:
        """

        def gaussianNoisy(im, mean, sigma):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        # 将图像转化成数组
        img = np.asarray(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        try:
            img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
            img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
            img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
            img[:, :, 0] = img_r.reshape([width, height])
            img[:, :, 1] = img_g.reshape([width, height])
            img[:, :, 2] = img_b.reshape([width, height])
        except:
            img = img
        return Image.fromarray(np.uint8(img))

    @staticmethod
    def gaussianNoisy(img, mean, sigma):
        img_arr = np.asarray(img)
        new = img_arr + np.random.normal(mean, sigma)
        return Image.fromarray(np.uint8(new))



    @staticmethod
    def saveImage(image, path):
        try:
            image.save(path)
        except:
            print('not save img: ', path)
            pass


files = []


def get_files(dir_path):
    global files
    if os.path.exists(dir_path):
        parents = os.listdir(dir_path)
        for parent in parents:
            child = os.path.join(dir_path, parent)
            if os.path.exists(child) and os.path.isfile(child):
                # child = child.split('/')[4:]
                # str_child = '/'.join(child)
                files.append(child)
            elif os.path.isdir(child):
                get_files(child)
        return files
    else:
        return None


if __name__ == '__main__':
    times = 1  # 重复次数
    imgs_dir = '/home/huadian/图片/12'
    new_imgs_dir = '/home/huadian/图片/13/'
    # if os.path.exists(new_imgs_dir):
    #    shutil.rmtree(new_imgs_dir)
    funcMap = {"flip": DataAugmentation.randomFlip,
               "rotation": DataAugmentation.randomRotation,
               "crop": DataAugmentation.randomCrop,
               "color": DataAugmentation.randomColor,
               "gaussian": DataAugmentation.randomGaussian
               }
    funcLists = {"flip", "rotation", "crop", "color", "gaussian"}
    # funcLists = {"flip", "rotation", "crop", "gaussian"}
    # funcLists = {"rotation",  "crop", "gaussian"}

    global _index
    imgs_list = get_files(imgs_dir)
    for index_img, img in enumerate(imgs_list):
        if index_img != 0 and index_img % 50 == 0:
            print('now is dealing %d image' % (index_img))
        tmp_img_dir_list = img.split('/')[:-1]
        tmp_img_dir_list[0:len(new_imgs_dir.split('/'))] = new_imgs_dir.split('/')
        new_img_dir = '/'.join(tmp_img_dir_list)

        if not os.path.exists(new_img_dir):
            os.makedirs(new_img_dir)
        try:
            shutil.copy(img, os.path.join(new_img_dir, img.split('/')[-1]))
        except:
            pass

        img_name = img.split('/')[-1].split('.')[0]
        postfix = img.split('.')[1]  # 后缀
        if postfix.lower() in ['jpg', 'jpeg', 'png', 'bmp']:
            image = DataAugmentation.openImage(img)
            _index = 1
            for func in funcLists:
                if func == 'flip':
                    flip_model = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
                    for model_index in range(len(flip_model)):
                        new_image = DataAugmentation.randomFlip(image, flip_model[model_index])
                        img_path = os.path.join(new_img_dir, img_name + '_' + str(_index) + '.' + postfix)
                        DataAugmentation.saveImage(new_image, img_path)
                        _index += 1
                elif func == 'gaussian':
                    new_image = DataAugmentation.randomGaussian(image)
                    img_path = os.path.join(new_img_dir, img_name + '_' + str(_index) + '.' + postfix)
                    DataAugmentation.saveImage(new_image, img_path)
                    _index += 1
                else:
                    for _i in range(0, times, 1):
                        new_image = funcMap[func](image)
                        img_path = os.path.join(new_img_dir, img_name + '_' + str(_index) + '.' + postfix)
                        DataAugmentation.saveImage(new_image, img_path)
                        _index += 1
