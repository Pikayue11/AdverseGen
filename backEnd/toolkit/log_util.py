import numpy as np
from PIL import Image

def saveImage(advImage, filename, extension, type): ## type: ori, adv, diff
    folder_name = './tmp/'
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

def getFileNameAndExtension(path):
    slashIndex = path.rindex('/')
    if(slashIndex == -1 or slashIndex+1 == len(path)):
        print('Please choose a right file path')
        return None, None
    extensionIndex = path.rindex('.')
    if (extensionIndex == -1 or extensionIndex + 1 == len(path) or extensionIndex == slashIndex + 1):
        print('Please choose a right file path with a correct extension')
        return None, None
    extention = path[extensionIndex + 1:]
    fileName = path[slashIndex + 1:extensionIndex]
    return fileName, extention


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



# three ugly function



def mapLabel(database, index):
    cifar_10_labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    cifar_10_labels = np.array(cifar_10_labels)

    if database == 'CIFAR-10':
        return cifar_10_labels[int(index)]
    else:
        print("mapLabel, not cifar 10, in log_util")
        return ''

from skimage.metrics import _structural_similarity as ss
def getNormValue(img, advImg, norm):
    if norm == 'SSIM':
        return compute_ssim(img, advImg)
    return -1

def compute_ssim(img1, img2): # [32,32,3]
    img1 = img1.copy().astype(np.uint8)
    img2 = img2.copy().astype(np.uint8)
    if img1.shape[2] > 1:
        return ss.structural_similarity(img1, img2, multichannel=True)
    else:
        return ss.structural_similarity(img1, img2, multichannel=False)


import torch
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2471, 0.2435, 0.2616]),
])
device = ('cuda' if torch.cuda.is_available() else 'cpu')
def getCons(model, img):
    if np.max(img) > 1:
        img = img / 255
    output = model(np.expand_dims(img, axis=0))
    output = torch.nn.Softmax(dim=1)(torch.from_numpy(output)).squeeze().numpy()

    return output

















