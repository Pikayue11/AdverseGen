import numpy as np
from .PQP import PQP

def attack_classifier(forward, model, ori_label, data_generator, iter_time,hard_attack=True, loss_goal=0.9, N=20):
    query_fun = lambda img: forward(model, img)
    mean = lambda x: np.asarray(x).mean()
    success, ssim, psnr, NQ = [], [], [], []
    adv_images = np.copy(data_generator)
    num_imgs = data_generator.shape[0]
    for i in range(num_imgs):
        img = data_generator[i]
        label = ori_label[i]
        probs = forward(model, img)
        loss_goal_ = loss_goal
        preds = np.argsort(probs)
        if hard_attack:
            target = preds[0]  # last predicted class
            print('*** Attacking image %d, original label %d, target label %d ***' % (i+1, label, target))
            print_every = iter_time *10
        else:
            target = preds[-2] # 2nd predicted class
            print('*** Attacking image %d, original label %d, target label %d ***' % (i+1, label, target))
            print_every = iter_time

        # start attack
        newImg, success_, ssim_, psnr_, NQ_, _ = PQP(int(ori_label[i]) ,query_fun=query_fun, or_img=img, target=target, loss_goal=loss_goal_, N=N,
                                                minimize_loss=False, print_every=print_every)
        newImg = np.uint8(newImg)
        adv_images[i] = newImg

        if(success_):
            success.append(1)
        else:
            success.append(0)
        ssim.append(ssim_)
        psnr.append(psnr_)
        NQ.append(NQ_)
    print('\n***Ending**')
    print('*** Success %d/%d, average ssim: %0.3f'
            % (sum(success), num_imgs, mean(ssim)))
    return adv_images, ssim, success