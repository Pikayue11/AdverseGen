from PIL import Image
import numpy as np
import L0_attack.L0_API as l0
import os
import time

runningFlag = 0

# to change the user upload image from jpg/jpeg into pngs format
# becase simpleGUI can only display .png or .gif images
def savePng(path, width, height):
    if os.path.exists(path) == False:
        return ''
    folder_name = 'pngs/'
    file_name = os.path.basename(path)
    if file_name[-3:].lower() == 'jpg':
        file_name = file_name[:-3]
    elif file_name[-4:].lower() == 'jpeg':
        file_name = file_name[:-4]
    elif file_name[-3:].lower() == 'png':
        file_name = file_name[:-3]
    else:
        return ''
    file_name += 'png'
    im = Image.open(path)
    im = im.resize((width,height))
    newPath = folder_name + file_name
    im.save(newPath)
    return newPath

def UpDimension(threeDImage):   # work for single image
    return np.expand_dims(threeDImage, axis=0)

def getImage(path):
    ori_image = Image.open(path)
    ori_image = np.array(ori_image)
    return UpDimension(ori_image)

def AE_L0(images):     # work for single image
    _, new_images, new_labels, L0_norms, success = l0.L0_api(images)
    im = Image.fromarray(new_images[0])
    image_path = 'AdvResults/new_test1.png'
    im.save(image_path)
    return image_path, new_labels[0], L0_norms[0], success[0]

def getLabel(images):
    images = l0.data_preprocess(images)
    return l0.get_labels(images,model = l0.load_model_L0())[0]


def getAdvPath(ori_images, imageInfo, window1):
    adv_newPath, adv_label_id, norm, success = AE_L0(ori_images)
    adv_label_name = imageInfo.labels[int(adv_label_id)]
    print("The label of the original image is", adv_label_name)
    print("The norm value is: ", norm)

    window1['-t6-'].update('Status: free            ')
    window1['-adv_image-'].update(size=(imageInfo.width, imageInfo.height), filename=adv_newPath)

def updateRunning(window1):
    status = ['States: running   ', 'States: running.  ', 'States: running.. ', 'States: running...']
    cnt = 1
    while window1['-t6-'].get()[0:18] in status:
        window1['-t6-'].update(status[cnt])
        cnt = cnt + 1
        cnt = cnt % 4
        time.sleep(1)



class ImageInfo():
    def __init__(self,database):
        self.database = database
        self.width, self.height = self.update_resolution(database)
        self.labels = self.update_labels(database)

    def update_resolution(self,database):
        if database == 'CIFAR-10':
            return 32, 32
        return -1, -1

    def update_labels(self,database):
        if database == 'CIFAR-10':
            return ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        return ['wky0','wky1','wky2']