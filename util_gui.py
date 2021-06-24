from PIL import Image
import numpy as np
import L0_attack.L0_API as l0
from backEnd.attacker import ImageAttacker
import os
import time
import io
import base64

class ImageInfo():
    def __init__(self, database):
        self.database = database
        self.resolution = self.update_resolution(database)

    def set_database(self, database):
        self.database = database

    def update_resolution(self, database):
        if database.lower() == 'cifar-10':
            return (32, 32)
        if database.lower() == 'imagenet':
            return (299, 299)
        return (-1, -1)

    def mapLabel(self, index):
        prefix = self.database.lower()
        fileName = './database/' + prefix + '_label'
        f = open(fileName, 'r')
        str = f.readlines()[int(index)]
        f.close()
        return str.split(',')[0].strip()

    def getLabelArray(self):
        prefix = self.database.lower()
        fileName = './database/' + prefix + '_label'
        f = open(fileName, 'r')
        str = f.readlines()
        str = [i.strip() for i in str]
        str = [i.split(',')[0] for i in str]
        f.close()
        return str

# to change the user upload image from jpg/jpeg into pngs format
# becase simpleGUI can only display .png or .gif images
# newPath: image/tmp/a.png
# long_name: a.png
def savePng(path, resolution, folder = 'images/tmp/', prefix = 'ori', new_extension = '.png'):
    if os.path.exists(path) == False:
        return '', ''
    file = os.path.basename(path)
    index = file.rfind('.')
    if index == -1:
        return '', ''
    file_name = file[:index]
    extension = file[index:]
    if extension.lower() not in ('.jpg', '.png', '.jpeg'):  # if it is in image format
        return '', ''
    long_name = prefix + '_' +file_name + new_extension
    newPath = folder + long_name
    im = Image.open(path)
    im = im.resize(resolution)
    im.save(newPath)
    return newPath, long_name

def UpDimension(threeDImage):  # work for single image
    return np.expand_dims(threeDImage, axis=0)

def getImage(path, resolution):
    ori_image = Image.open(path)
    ori_image = ori_image.resize(resolution)
    ori_image = np.array(ori_image)
    return UpDimension(ori_image)

def AE_L0(images):  # work for single image
    _, new_images, new_labels, L0_norms, success = l0.L0_api(images)
    im = Image.fromarray(new_images[0])
    image_path = 'AdvResults/new_test1.png'
    im.save(image_path)
    return image_path, new_labels[0], L0_norms[0], success[0]

def getAdvPath(attacker: ImageAttacker, ori_image, label, imageInfo: ImageInfo, window1, target_label=None):
    input = ori_image / 255
    adv_image, adv_label_id, norm, success = attacker.run(input, label, target_label, window1['-ev-'].get())
    img = (adv_image * 255).astype(np.uint8)
    adv_label_name = imageInfo.mapLabel(adv_label_id)
    im = Image.fromarray(img[0])
    # image_path = 'AdvResults/new_test1.png'
    # im.save(image_path)

    window1['-t3-'].update('Status: free            ')
    im_zoom = convert_to_bytes(im, (200, 200))
    window1['-adv_image-'].update(data=im_zoom)
    pert = np.squeeze(adv_image - input)
    pert_value = np.mean(np.abs(pert))
    pert -= np.min(pert)
    pert /= np.max(pert)
    pert_img = Image.fromarray((pert * 255).astype(np.uint8))
    pert_img_zoom = convert_to_bytes(pert_img, (200, 200))
    window1['-pert_image-'].update(data=pert_img_zoom)
    window1['-adv_image-'].update(data=im_zoom)
    window1['-pert_value-'].update('modified pixels avg: %.4f' % pert_value)
    window1['-adv_label-'].update(f'label: {imageInfo.mapLabel(adv_label_id[0])}')


def updateRunning(window1):
    status = ['States: running   ', 'States: running.  ', 'States: running.. ', 'States: running...']
    cnt = 1
    pert_image_zoom, adv_image_zoom = None, None
    while window1['-t3-'].get()[0:18] in status:
        # update mid process
        if os.path.exists(r'./images/tmp/diff_wkyTest.png'):
            pert_image_zoom = convert_to_bytes(r'./images/tmp/diff_wkyTest.png', (200, 200))
            adv_image_zoom = convert_to_bytes(r'./images/tmp/adv_wkyTest.png', (200, 200))
        window1['-pert_image-'].update(data=pert_image_zoom)
        window1['-adv_image-'].update(data=adv_image_zoom)
        window1['-t3-'].update(status[cnt])
        cnt = cnt + 1
        cnt = cnt % 4
        time.sleep(1)

def convert_to_bytes(file_or_bytes, resize=None):
    '''
    Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
    Turns into  PNG format in the process so that can be displayed by tkinter
    :param file_or_bytes: either a string filename or a bytes base64 image object
    :type file_or_bytes:  (Union[str, bytes])
    :param resize:  optional new size
    :type resize: (Tuple[int, int] or None)
    :return: (bytes) a byte-string object
    :rtype: (bytes)
    '''
    if isinstance(file_or_bytes, Image.Image):
        img = file_or_bytes
    else:
        if isinstance(file_or_bytes, str):
            img = Image.open(file_or_bytes)
        else:
            try:
                img = Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
            except Exception as e:
                dataBytesIO = io.BytesIO(file_or_bytes)
                img = Image.open(dataBytesIO)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height / cur_height, new_width / cur_width)
        img = img.resize((int(cur_width * scale), int(cur_height * scale)), Image.ANTIALIAS)
    with io.BytesIO() as bio:
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
