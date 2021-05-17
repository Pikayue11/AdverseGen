import os
import random
import numpy as np
from skimage.metrics import _structural_similarity as ss
from SSIM_grad import build_ssim_graph

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def PQP_inner(get_loss_and_gain, img, loss, low_ssim_grad_pixels, N=20, delta=1, k_max=20):
    gain = -1
    k = 0
    while gain < 0 and k < k_max:

        # build the perturbation matrix W by randomly picking N low-ssim-gradient pixels
        random.shuffle(low_ssim_grad_pixels)
        W = np.zeros(img.shape[:2] + (1,), np.float32)  # W broadcast along color channels
        for i in range(N):
            W[low_ssim_grad_pixels[i]] = random.randint(0, 1) * 2 - 1

        # + delta
        img_plus = (img + delta * W).clip(0., 255.)
        loss_plus, gain_plus = get_loss_and_gain(img_plus, loss)

        # - delta
        img_minus = (img - delta * W).clip(0., 255.)
        loss_minus, gain_minus = get_loss_and_gain(img_minus, loss)

        k += 1

        # reject condition
        if gain_plus < 0 and gain_minus < 0 and k < k_max:
            continue

        # pick the best one
        if gain_plus >= gain_minus:
            img = img_plus
            loss = loss_plus
            gain = gain_plus
        else:
            img = img_minus
            loss = loss_minus
            gain = gain_minus

    return img, loss, k

def PQP(or_label, query_fun, or_img, target, loss_goal=None, minimize_loss=False, ssim_th=0.95, M=66., N=20, delta=1, k_max=20, print_every=100):
    """
    Perform Perceptual Quality Preserving (PQP) black-box attack
    :param query_fun: a function which takes as input:
                        an image (3-D channel-last numpy array in the range [0, 255]) and
                        a target (either an integer scalar representing the target class or a numpy feature vector to approximate)
                        and returns the black-box loss (either to minimize or maximize)
    :param or_img: the original image in the format accepted by query_fun
    :param target: the target in the format accepted by query_fun
    :param loss_goal: the loss value to reach in order to consider the attack successful in the format returned by query_fun
                      [default: if the target is a scalar integer, the default loss_goal value is 0.9]
    :param minimize_loss: a boolean value to decide whether to minimize or maximize the loss [default: False, i.e. maximize]
    :param ssim_th: the SSIM value at which the attack is considered failed [default: 0.95]
    :param M: the percentage of spatial pixels with lowest SSIM gradient to consider as candidates [default: 66.0]
    :param N: the amount of pixels to perturb at each query [default: 20]
    :param delta: the strength of the perturbation [default: 1.0]
    :param k_max: the number of tentative before to accept a perturbation that does not improve the attack [default: 20]
    :param print_every: print log every n iterations [default: 100]
    :return: the attacked image and the attack quality metrics:
             if the attack succeed (bool),
             the perceptual quality of the attacked image in terms of SSIM (float),
             the quality of the attacked image in terms of PSNR in dB (float),
             the number of queries needed (int),
             and the final loss achieved (float)
    """

    assert or_img.ndim == 3 and (or_img.shape[2] == 3 or or_img.shape[2] == 1), 'or_img must be a 3-D channel-last numpy array.'
    if loss_goal is None:
        if isinstance(0, int) or np.issubdtype(target, np.integer):
            loss_goal = 0.9
            print('loss_goal = 0.9')
        else:
            raise ValueError('If the target is not a class, the loss_goal parameter must be passed to the PQP function.')
    att_img = or_img.copy()
    grad_ssim_fun_, tf_sess = build_ssim_graph((1, ) + or_img.shape)
    grad_ssim_fun = lambda x, y=or_img: grad_ssim_fun_(x, y)

    M = int(float(M) * or_img.shape[-1] * or_img.shape[-2])


    get_for = lambda im: query_fun(np.round(im).clip(0., 255.))
    if minimize_loss:
        def get_gain(loss, t=loss_goal):
            return t - loss
    else:
        def get_gain(loss, t=loss_goal):
            return loss - t

    def get_loss_and_gain(im, t=loss_goal):
        loss = get_for(im)[target]
        gain = get_gain(loss, t)
        return loss, gain

    def compute_psnr(img1, img2=or_img):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100.
        else:
            return 10 * np.log10(255.0 * 255.0 / mse)

    def compute_ssim(img1, img2=or_img):
        img1 = img1.copy().astype(np.uint8)
        img2 = img2.copy().astype(np.uint8)
        if img1.shape[2] > 1:
            return ss.structural_similarity(img1, img2, multichannel=True)
        else:
            return ss.structural_similarity(img1, img2, multichannel=False)

    ssim = compute_ssim(att_img)
    psnr = compute_psnr(att_img)
    loss, gain = get_loss_and_gain(att_img)


    NQ = 0
    iter = 0
    cons = get_for(att_img)
    print('[log][Start       ] origin: %0.3f target: %0.3f ssim ssim: %0.3f' % (cons[or_label], cons[target], ssim))

    while gain < 0 and ssim >= ssim_th:
        # compute the SSIM gradient and get the first M smallest pixel locations
        ssim_grad = grad_ssim_fun(att_img)
        low_ssim_grad_pixels = np.argsort(np.reshape(ssim_grad, -1), 0)
        low_ssim_grad_pixels = [np.unravel_index(i, ssim_grad.shape) for i in low_ssim_grad_pixels[0:M]]

        att_img, loss, k = PQP_inner(get_loss_and_gain, att_img, loss, low_ssim_grad_pixels, N, delta, k_max)
        gain = get_gain(loss)
        NQ += (k * 2)

        iter += 1
        cons = get_for(att_img)
        if cons.argmax() == target:
            print('[log][iter %7d] origin: %0.3f target: %0.3f ssim ssim: %0.3f' % (iter, cons[or_label], cons[target], ssim))
            break
        if iter % print_every == 0:
            ssim = compute_ssim(or_img, att_img)
            psnr = compute_psnr(or_img, att_img)
            print('[log][iter %7d] origin: %0.3f target: %0.3f ssim ssim: %0.3f'  % (iter, cons[or_label], cons[target], ssim))

    cons = get_for(att_img)
    success = (cons.argmax() == target)
    ssim = compute_ssim(or_img, att_img)
    psnr = compute_psnr(or_img, att_img)
    if success:
        print('[log][End: success] original label %d, new label %d, ssim=%0.3f\n' % (or_label, cons.argmax(), ssim))
    else:
        print('[log][End: failed ] original label %d, new label %d, ssim=%0.3f\n' % (or_label, cons.argmax(), ssim))
    tf_sess.close()
    return att_img, success, ssim, psnr, NQ, loss



