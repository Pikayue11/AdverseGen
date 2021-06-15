import torch
from PIL import Image
import numpy as np
from resnet import ResNet18
import backEnd.toolkit as tk

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
# classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def load_model_L0():
    model = ResNet18().to(device)
    if torch.cuda.is_available():
        ckpt = torch.load('./model_weight/model_test.pt')
    else:
        ckpt = torch.load('./model_weight/model_test.pt', map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    return model

images = np.zeros((1,32,32,3))
img_dir1 = './images/5_16.jpg'  # dog 5
img1 = Image.open(img_dir1)
img1 = np.array(img1)
images[0,:,:,:] = img1
images = images / 255

images_torch = torch.from_numpy(images).permute(0, 3, 1, 2).float()
model = load_model_L0()
with torch.no_grad():
    y_test = model(images_torch.to(device))
print(y_test)

fmodel = tk.PyTorchModel(model, bounds=(0, 1))
attack = tk.attacks.HopSkipJump()
epsilons = [1.0]
_, advs, success = attack(fmodel, images_torch, torch.tensor([5]), epsilons=epsilons)
print(success)





