import os
import subprocess
import time
import torch
import torch.optim as optim
from skimage.io import imread, imsave

from .Learning_black_box.data_utils import *
from .Learning_black_box.LeBA10 import get_preprocess, load_images_data, QueryModel, run_attack_train
from .Learning_black_box.imagenet import get_model

from .base import MinimizationAttack
from ..distances import l2
from ..criteria import TargetedMisclassification

'''
img: png format, 0-255
'''
def get_device(device):
    import torch

    if device is None:
        return torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device

def check_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_single_img_attack(model, img, label, target=None, logger=None, **kwargs):
    temp_path = "LeBA_single_img"
    output_dir = "LeBA_single_img_output"
    # save img and label into LeBA folder
    check_mkdir(temp_path)
    imsave(temp_path + "/img.png", img[0])
    with open(temp_path + "/label", 'w') as f:
        f.write("img.png %d" % label)
        if target is not None:
            f.write(" %d" % target)
    # call LeBA method to run
    check_mkdir(output_dir)
    check_mkdir(output_dir + "/slog")
    # subprocess.run('cp %s %s/' % ("LeBA10.py", output_dir), shell=True)
    # subprocess.run('cp %s %s/' % ("run_single_img_attack.py", output_dir), shell=True)
    # cmd = get_cmd(1, '', 'SimBA++', victim_model, 'resnet152', temp_path, 'label', output_dir,
    #               '', 0)
    # run_cmd(cmd, output_dir, 'LeBA_images')

    # Set random seed

    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # random.seed = seed

    # Get victim model(model1) and surrogate model(model2) and wrap them for multi gpus.
    cpu_model2 = get_model('resnet152')

    data_loader = load_images_data(temp_path, 1, False, 'label')

    # # gpu id
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # gpu_num = len(args.gpu_id.split(','))
    # if gpu_num == 0:
    #     gpu_num = 1
    # device = torch.device("cuda")
    # model = nn.DataParallel(cpu_model.to(device), device_ids=[i for i in range(gpu_num)])
    # model2 = nn.DataParallel(cpu_model2.to(device), device_ids=[i for i in range(gpu_num)])
    device = get_device(None)
    model2 = cpu_model2.to("cuda:1")
    model2.eval()

    # Set output dir
    out_dir = output_dir  # used to be try20
    check_mkdir(out_dir + '/images')
    check_mkdir(out_dir + '/snapshot')
    check_mkdir(out_dir + '/logs')
    check_mkdir(out_dir + '/gauss_images')

    # preprocess functions for model1, model2
    preprocess1 = get_preprocess(model)
    preprocess2 = get_preprocess(model2)

    # query functions for model1, model2
    query = QueryModel("", model)
    query2 = QueryModel('', model2).query

    optimizer = 0
    b = 0

    mode = 'SimBA'
    pretrain_weight = ''
    batch_size = 0
    task_id = 1

    if_train = False
    with_TIMI = True
    with_s_prior = True

    if mode == 'train':  # LeBA
        if_train = True
        minibatch = 8
    elif mode == 'test':  # LeBA test mode
        minibatch = 8
        if pretrain_weight == '':
            pretrain_weight == 'this_weight'
    elif mode == 'SimBA++':  # SimBA++
        minibatch = 8
        pretrain_weight = ''
    elif mode == 'SimBA+':
        minibatch = 8
        pretrain_weight = ''
        with_TIMI = False
    elif mode == 'SimBA':
        minibatch = 16
        with_TIMI = False
        with_s_prior = False

    if batch_size != 0:
        minibatch = batch_size

    if mode[:3] != 'all':
        log_name = "log_" + mode + '_' + temp_path.split('/')[-1] + '_idx%d_0' % (
            task_id)  # result file name

        if pretrain_weight == 'this_weight':  # Load last trained surrogate weight
            model2.load_state_dict(torch.load(out_dir + '/snapshot/' + model2 + '_final.pth'))
        elif pretrain_weight != '':
            model2.load_state_dict(torch.load(pretrain_weight))

        data_iter = iter(data_loader)
        optimizer = optim.SGD(model2.parameters(), lr=0.005, momentum=0.9)


        # Run LeBA
        adv_img, counts_all, correct_all, end_type_all, L2_all = run_attack_train(model, model2, data_loader, minibatch,
                                                                         preprocess1, preprocess2, optimizer,
                                                                         log_name=log_name, out_dir=out_dir, logger=logger,
                                                                         if_train=if_train, with_TIMI=with_TIMI,
                                                                         with_s_prior=with_s_prior)
    # delete temporary files
    os.remove(temp_path+"/img.png")
    os.remove(temp_path+"/label")
    os.rmdir(temp_path)

    adv_img = torch.from_numpy(adv_img / 255)
    return adv_img


class LeBA(MinimizationAttack):
    distance = l2
    def __init__(self):
        self.distance = l2

    def run(self, model, inputs, criterion, logger=None, **kwargs):
        model.set_type('Pytorch')
        if isinstance(criterion, TargetedMisclassification):
            adv_img = run_single_img_attack(model, inputs, criterion.labels.raw, target=criterion.target_classes.raw, logger=logger)
        else:
            adv_img = run_single_img_attack(model, inputs, criterion.labels.raw, logger=logger)
        adv_img = adv_img.unsqueeze(0)
        model.set_type('Numpy')
        return adv_img.float().numpy()

