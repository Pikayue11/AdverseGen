import torch
import numpy as np
from resnet import ResNet18
from cornersearch_attacks_pt import CSattack

device = ('cuda' if torch.cuda.is_available() else 'cpu')
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

def data_preprocess(image_arr):     # images:  [ n, 32 ,32 , 3]
    return image_arr/255.0

def get_labels(trans_images, model):
    x_test_t = torch.from_numpy(trans_images).permute(0, 3, 1, 2).float()
    with torch.no_grad():
        y_test = model(x_test_t.to(device))
    y_test = y_test.cpu().numpy()
    labels = np.zeros((trans_images.shape[0]))
    for i in range(trans_images.shape[0]):
        labels[i] = y_test[i].argmax()
    return labels

def get_con(one_trans_image, model, ori_label): # 1, 32, 32, 3; get the max confidence of one image
    x_test_t = torch.from_numpy(one_trans_image).permute(0, 3, 1, 2).float()
    with torch.no_grad():
        y_test = model(x_test_t.to(device))
    cons = torch.nn.Softmax(dim=1)(y_test)

    print(cons)
    origin_con = cons[0][ori_label]
    max_index = cons[0].argmax()
    max = cons[0].max()
    cons[0][max_index] = 0

    second_max = cons[0].max()
    second_max_index = cons[0].argmax()

    cons[0][max_index] = max
    return origin_con, second_max, second_max_index

def run_loop_attack(model, trans_images, ori_labels):
    getCon = lambda img, ori_label : get_con(img, model, ori_label)
    args = {'type_attack': 'L0',
            'n_iter': 1000,
            'n_max': 100,
            'kappa': -1,
            'epsilon': -1,
            'sparsity': 10,
            'size_incr': 1}
    attack = CSattack(model, args)
    new_trans_images, L0_norms = attack.perturb(getCon, trans_images, ori_labels)
    return new_trans_images, L0_norms

def recover_images(trans_images):
    images = trans_images * 255
    images = images.astype(np.uint8)
    return images

def get_success(label1, label2):    # non-target
    n = label1.shape[0]
    success = np.zeros((n))
    for i in range(n):
        if label1[i] != label2[i]:
            success[i] = 1
    return success

def L0_api(images_arr):  # images:  [ n, 32 ,32 , 3]
    # load model, resnet18, pretrained on cifar-10
    model = load_model_L0()

    # process data and save them into x_test [n,32,32,3], save the label of x_test in y_test [n]
    trans_images = data_preprocess(images_arr)


    # get original label
    ori_labels = get_labels(trans_images, model)

    # run attack, adv save the trans_images after attack
    new_trans_images, L0_norms = run_loop_attack(model, trans_images, ori_labels)

    # the labels after attack
    new_labels = get_labels(new_trans_images, model)

    # the Adversarial Examples
    new_images = recover_images(new_trans_images)

    # if a picture's attack is success, the success will be 1
    success = get_success(ori_labels,new_labels)

    return ori_labels, new_images, new_labels, L0_norms, success







