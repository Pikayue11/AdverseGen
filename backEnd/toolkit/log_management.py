from .log_util import *
import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)

class LogManagement():
    def __init__(self,
                 img,
                 oriLabel,
                 modelwithName,  # [model, name]
                 distance,
                 old_norm,
                 imageName='testImage',
                 target=None,
                 databaseName='CIFAR-10',
                 extention='png'
                 ):
        self.imageName = imageName
        self.extention = extention
        self.model = modelwithName[0]
        self.modelName = modelwithName[1]
        self.databaseName = databaseName
        self.oriLabel = oriLabel
        self.oriLabelStr = mapLabel(self.databaseName, self.oriLabel)
        self.target = None
        if target is None:
            self.isTarget = False
        else:
            self.isTarget = True
            self.target = target
        self.distance = distance
        self.img = img
        self.norm = self.updateNorm(old_norm)
        self.logStart()

    def updateNorm(self, old_norm):
        new_norm = {}
        map_formal = {'l0': 'L0 distance', 'l2': 'L2 distance', 'l8': 'Linf distance', 'ssim': 'SSIM'}
        for i in old_norm:
            new_norm[map_formal[i]] = old_norm[i]
        return new_norm

    def logStart(self):
        attackType = 'target' if self.isTarget else 'non-target'
        targetMsg = '    Target label: %s\n' % (mapLabel(self.databaseName, self.target)) if self.isTarget else ''
        logger.info('\n*** \n    Launching an attack to image %s\n' % (self.imageName) +
                    '    Constraints: %s\n' % (self.norm) +
                    '    Attack type: %s\n' % (attackType) +
                    '    Original label: %s\n' % (self.oriLabelStr) + targetMsg +
                    '    Model: %s\n' % (self.modelName) +
                    '    Database: %s\n' % (self.databaseName) +
                    '***')
        saveImage(self.img[0], self.imageName, self.extention, 'ori')
        saveImage(self.img[0] - self.img[0], self.imageName, self.extention, 'diff')
        saveImage(self.img[0], self.imageName, self.extention, 'adv')

    def imgUpdate(self, advImg, iter=-1):
        if self.isTarget:
            self.updateTarget(advImg, iter)
        else:
            self.updateNonTarget(advImg, iter)
        saveImage(advImg[0], self.imageName, self.extention, 'adv')
        saveImage(get_diff(self.img[0], advImg[0]), self.imageName, self.extention, 'diff')

    def updateTarget(self, advImg, iter):
        cons = getCons(self.model, advImg[0])
        normValue = getNormValue(self.img, advImg, self.distance)
        pre_norm = norm_presentation(self.norm, normValue)
        if (iter <= 0):
            logger.info('origin_con: %.4f, target_con: %.4f, Norm: %s' % (
            cons[int(self.oriLabel)], cons[int(self.target)], pre_norm))
        else:
            logger.info('[Iter: %7d] origin_con: %.4f, target_con: %.4f, Norm: %s' % (
            iter, cons[int(self.oriLabel)], cons[int(self.target)], pre_norm))

    def updateNonTarget(self, advImg, iter):
        cons = getCons(self.model, advImg[0])
        normValue = getNormValue(self.img, advImg, self.distance)

        pre_norm = norm_presentation(self.norm, normValue)

        if (iter <= 0):
            logger.info('origin_con: %.4f, Norm: %s' % (cons[self.oriLabel],  pre_norm))
        else:
            logger.info('[Iter: %7d] origin_con: %.4f, Norm: %s' % (iter, cons[self.oriLabel], pre_norm))

    def logEnd(self, advImg):
        if self.isTarget:
            self.endTarget(advImg)
        else:
            self.endNonTarget(advImg)
        saveImage(advImg[0], self.imageName, self.extention, 'adv')
        saveImage(get_diff(self.img[0], advImg[0]), self.imageName, self.extention, 'diff')

    def endTarget(self, advImg):
        cons = getCons(self.model, advImg[0])
        normValue = getNormValue(self.img, advImg, self.distance)
        pre_norm = norm_presentation(self.norm, normValue)
        success = 'successed' if cons.argmax() == self.target else 'failed'
        logger.info('\n*** \n    End of the attack, it %s\n'

                    '    Origin_label: %s\n'
                    '    Target_label: %s\n'
                    '    Current label: %s\n'
                    '    Norm: %s\n'
                    '***' % (success, self.oriLabelStr,
                             mapLabel(self.databaseName, self.target),
                             mapLabel(self.databaseName, cons.argmax()),
                             pre_norm))

    def endNonTarget(self, advImg):
        cons = getCons(self.model, advImg[0])
        normValue = getNormValue(self.img, advImg, self.distance)
        pre_norm = norm_presentation(self.norm, normValue)
        success = 'successed' if cons.argmax() != self.oriLabel else 'failed'
        logger.info('\n*** \n    End of the attack, it %s\n'
                    '    Origin_label: %s\n'
                    '    Current label: %s\n'
                    '    Norm: %s\n'
                    '***' % (success, self.oriLabelStr,
                             mapLabel(self.databaseName, cons.argmax()),
                             pre_norm))
