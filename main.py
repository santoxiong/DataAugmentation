import os
import random
from F import DataAugmentation as FD

absolute_path = os.getcwd()
img_path = os.path.join(absolute_path, 'IMG')
save_path = os.path.join(absolute_path, 'IMGS')

images = os.listdir(img_path)

for image in images:
    name = image.split('.')[0]
    img_dir = os.path.join(img_path, image)

    image = FD.openImage(img_dir)
    w, h = image.size

    path0 = os.path.join(save_path, f'{name}xs.png')
    path1 = os.path.join(save_path, f'{name}s.png')
    path2 = os.path.join(save_path, f'{name}m.png')
    path3 = os.path.join(save_path, f'{name}l.png')
    path4 = os.path.join(save_path, f'{name}xl.png')

    sigma0 = round(random.uniform(0.1, 0.3), 2)
    sigma1 = round(random.uniform(0.3, 0.5), 2)
    sigma2 = round(random.uniform(0.5, 0.7), 2)
    sigma3 = round(random.uniform(0.7, 0.9), 2)
    sigma4 = round(random.uniform(0.9, 1.2), 2)

    new_image0 = image.resize((int(sigma0 * w), int(h)))
    new_image1 = image.resize((int(sigma1 * w), int(h)))
    new_image2 = image.resize((int(sigma2 * w), int(h)))
    new_image3 = image.resize((int(sigma3 * w), int(h)))
    new_image4 = image.resize((int(sigma3 * w), int(h)))

    FD.saveImage(new_image0, path0)
    FD.saveImage(new_image1, path1)
    FD.saveImage(new_image2, path2)
    FD.saveImage(new_image3, path3)
    FD.saveImage(new_image4, path4)
