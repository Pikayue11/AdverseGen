import torch
# import torchvision.datasets as datasets
# import torch.utils.data as data
# import torchvision.transforms as transforms
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as transforms
from resnet import ResNet18
from utils_pt import load_data

# classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Define hyperparameters.')
  parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, mnist')
  parser.add_argument('--attack', type=str, default='CS', help='PGD, CS')
  parser.add_argument('--path_results', type=str, default='none')
  parser.add_argument('--n_examples', type=int, default=2)
  parser.add_argument('--data_dir', type=str, default= './dataset_images')
  parser.add_argument('--img_dir', type=str, default='./images/5_16.jpg')
  
  hps = parser.parse_args()
  
  # load model
  device = ('cuda' if torch.cuda.is_available() else 'cpu')
  model = ResNet18().to(device)
  ckpt = torch.load('./models/model_test.pt',map_location = 'cpu')
  model.load_state_dict(ckpt)
  model.eval()
  
  # load data



  x_test, y_test = load_data(hps.dataset, hps.n_examples, hps.data_dir)
  # print(x_test.shape)
  # print(x_test)
  # x_test = np.squeeze(x_test,0)
  # im = Image.fromarray(x_test)
  # im.save('a.jpg')






  # # x_test, y_test are images and labels on which the attack is run
  # # x_test in the format bs (batch size) x heigth x width x channels
  # # y_test in the format bs

  trans = transforms.Compose([transforms.ToTensor()])

#
#   x_test_ori = Image.open(hps.img_dir)
#   print(x_test_ori)
#   x_test = trans(x_test_ori).numpy()
#   print(x_test)
#   print(x_test.shape)
#   x_test = np.expand_dims(x_test, 0)
#   # x_test_ori = np.array(x_test_ori)
#   # x_test_ori = np.expand_dims(x_test_ori, 0)
# # <PIL.Image.Image image mode=RGB size=32x32 at 0x19AE3702F40>
#   # x_test_t = torch.from_numpy(x_test).permute(0, 3, 1, 2).float()
#   x_test_t = torch.from_numpy(x_test).float()
#   print(x_test.shape)
#   print(x_test_t.shape)
#   with torch.no_grad():
#     y_test = model(x_test_t.to(device))
#
#   print(y_test)
#   y_test = y_test.argmax()
#   y_test = np.expand_dims(y_test, 0)
#   print(y_test)
#   x_test = torch.from_numpy(x_test).permute(0, 2, 3, 1).numpy()




  if hps.attack == 'PGD':
    import pgd_attacks_pt

    args = {'type_attack': 'L0',
                'n_restarts': 5,
                'num_steps': 100,
                'step_size': 120000.0/255.0,
                'kappa': -1,
                'epsilon': -1,
                'sparsity': 5}

    attack = pgd_attacks_pt.PGDattack(model, args)

    adv, pgd_adv_acc = attack.perturb(x_test, y_test)

    if hps.path_results != 'none': np.save(hps.path_results + 'results.npy', adv)

  elif hps.attack == 'CS':
    import cornersearch_attacks_pt

    args = {'type_attack': 'L0',
            'n_iter': 1000,
            'n_max': 100,
            'kappa': -1,
            'epsilon': -1,
            'sparsity': 10,
            'size_incr': 1}

    attack = cornersearch_attacks_pt.CSattack(model, args)

    adv, pixels_changed = attack.perturb(x_test, y_test)


    adv = np.array(adv)
    print(adv.shape)
    adv = np.squeeze(adv,0)
    print(adv.shape)


    adv = adv * 255

    # print(Image.fromarray(adv.astype(np.uint8)))
    im = Image.fromarray(adv.astype(np.uint8))
    im.save('a.jpg')

    # if hps.path_results != 'none': np.save(hps.path_results + 'results.npy', adv)

