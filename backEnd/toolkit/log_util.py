import numpy as np
from PIL import Image
import torch

def saveImage(advImage, filename, extension, type): ## type: ori, adv, diff
    folder_name = './images/tmp/'
    newName = folder_name + type + '_' + filename + '.' + extension
    im = denormalize_images(advImage, type)
    im = Image.fromarray(im)
    im.save(newName)

def denormalize_images(image, type):
    image = np.array(image)
    if type != 'diff':
        if(np.max(image) <= 1):
            image *= 255
    return image.astype(np.uint8)

def get_diff(img1, img2): #[32 ,32 ,3]
    img1 = np.array(img1)
    img2 = np.array(img2)
    diff_img = img1 - img2
    diff_img = diff_img - np.min(diff_img)
    if np.max(diff_img) > 0:
        diff_img = diff_img / np.max(diff_img)
        diff_img *= 255
    diff_img = np.array(diff_img)
    return diff_img.astype(np.uint8)

def mapLabel(database, index):
    prefix = database.lower()
    fileName = './database/' + prefix + '_label'
    f = open(fileName, 'r')
    str = f.readlines()[int(index)]
    f.close()
    return str.split(',')[0].strip()


def getNormValue(img, advImg, distance):
    norm_values = []
    for i in distance:
        norm_values.append(i(img, advImg)[0])
    return norm_values

def norm_presentation(norm, norm_value):
    cnt = 0
    pre_norm = {}
    for i in norm:
        pre_norm[i] = norm_value[cnt]
        cnt += 1
    return pre_norm

def getCons(model, img):
    if np.max(img) > 1:
        img = img / 255
    output = model(np.expand_dims(img, axis=0))
    output = torch.nn.Softmax(dim=1)(torch.from_numpy(output)).squeeze().numpy()

    return output





















