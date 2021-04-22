from PIL import Image
import os


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
    print(im.shape)
    newPath = folder_name + file_name
    im.save(newPath)
    return newPath


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