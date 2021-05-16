import torch
import numpy as np
from resnet import resnet18
from torchvision import transforms
from PQP_attack import attack_classifier

device = ('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2471, 0.2435, 0.2616]),
])

def load_model_ssim():
    model = resnet18(pretrained=False).to(device)
    if torch.cuda.is_available():
        ckpt = torch.load('./CIFAR10_pretrained_models/resnet18.pt')
    else:
        ckpt = torch.load('./CIFAR10_pretrained_models/resnet18.pt', map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    return model

def get_confidence(model, img): # img is [32,32,3], the elements are from [0 to 255]
    with torch.no_grad():
        # change img from [32,32,3] to [1,32,32,3]
        img = transform(img.astype(np.uint8)).unsqueeze(0).to(device)
        output = torch.nn.Softmax(dim=1)(model(img)).squeeze()
        output = output.data.cpu().numpy()
    return output

def get_label(model, image_arr):

    labels = np.zeros(image_arr.shape[0])
    for i in range(image_arr.shape[0]):
        labels[i] = get_confidence(model, image_arr[i]).argmax()
    return labels


def SSIM_api(image_arr):
    model = load_model_ssim()

    ori_labels = get_label(model, image_arr)

    new_images, ssim, success = attack_classifier(get_confidence, model, ori_labels, image_arr,10, hard_attack=False)

    new_labels = get_label(model, np.array(new_images))

    return ori_labels, np.uint8(new_images), new_labels, ssim, success







