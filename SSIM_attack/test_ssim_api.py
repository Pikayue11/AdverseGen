import numpy as np
import SSIM_API as ssim
from PIL import Image

N = 1
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



ori_labels, new_images, new_labels, L0_norms, success = ssim.SSIM_api(images)

print(ori_labels)
print(new_labels)
print(L0_norms)
print(success)

image_name = ['results/new_img1.jpg','results/new_img2.jpg']
for i in range(N):
    im = Image.fromarray(new_images[i])
    im.save(image_name[i])



# model = ssim.load_model_ssim()
# cons = ssim.get_confidence(model, img1)
#
