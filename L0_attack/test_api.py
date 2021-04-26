import L0_API as l0
from PIL import Image
import numpy as np

#N is the number of images
N = 2
images = np.zeros((N,32,32,3))

if N == 1:
    img_dir1 = './images/5_16.jpg'  # dog 5
    img1 = Image.open(img_dir1)
    img1 = np.array(img1)
    images[0,:,:,:] = img1
elif N == 2:
    img_dir1 = './images/5_16.jpg'  # dog 5
    img_dir2 = './images/8_1.jpg'  # ship 8
    img1 = Image.open(img_dir1)
    img2 = Image.open(img_dir2)
    img1 = np.array(img1)
    img2 = np.array(img2)
    images[0, :, :, :] = img1
    images[1,:,:,:] = img2


# images is a numpy array whose shape is [N, 32, 32, 3]
# This is the most import function, run about 2-3 minutes for 1 image *******
# cifar_10_classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

ori_labels, new_images, new_labels, L0_norms, success = l0.L0_api(images)

print(ori_labels)
print(new_labels)
print(L0_norms)
print(success)

image_name = ['results/new_img1.jpg','results/new_img2.jpg']
for i in range(N):
    im = Image.fromarray(new_images[i])
    im.save(image_name[i])





# not important test:  get labels

# model = l0.load_model_L0()
# trans_images = l0.data_preprocess(images)
# labels = l0.get_labels(trans_images, model)
# print(labels)