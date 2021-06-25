from .distances import *
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
formatter = logging.Formatter('%(asctime)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler(sys.stdout))

class LogManagement():
    def __init__(self,
                 img,
                 oriLabel,
                 modelwithName,  # [model, name]
                 norm,
                 target=None,
                 databaseName='CIFAR-10',
                 imageName='wkyTest',
                 extention='png'):
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
            self.target = target[0]
        self.norm = norm
        self.img = img
        self.logStart()

    def logStart(self):
        attackType = 'target' if self.isTarget else 'non-target'
        targetMsg = 'Target label: %s' % (mapLabel(self.databaseName, self.target)) if self.isTarget else ''
        logger.info('\n*** \n    Launching an attack to image %s\n' % (self.imageName) +
                    '    Norm: %s\n' % (self.norm) +
                    '    Attack type: %s' % (attackType) +
                    '    Original label: %s\n    ' % (self.oriLabelStr) + targetMsg +
                    '\n    Model: %s\n' % (self.modelName) +
                    '    Database: %s\n' % (self.databaseName) +
                    '***')
        print('\n*** \n    Launching an attack to image %s\n' % (self.imageName) +
              '    Norm: %s\n' % (self.norm) +
              '    Attack type: %s' % (attackType) +
              '    Original label: %s\n    ' % (self.oriLabelStr) + targetMsg +
              '\n    Model: %s\n' % (self.modelName) +
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
        normValue = getNormValue(self.img, advImg, self.norm)
        if (iter <= 0):
            logger.info('origin_con: %.4f, target_con: %.4f, %s: %.3f' % (
            cons[int(self.oriLabel)], cons[int(self.target)], self.norm, normValue))
            print('origin_con: %.4f, target_con: %.4f, %s: %.3f' % (
            cons[int(self.oriLabel)], cons[int(self.target)], self.norm, normValue))
        else:
            logger.info('[Iter: %7d] origin_con: %.4f, target_con: %.4f, %s: %.3f' % (
            iter, cons[int(self.oriLabel)], cons[int(self.target)], self.norm, normValue))
            print('[Iter: %7d] origin_con: %.4f, target_con: %.4f, %s: %.3f' % (
            iter, cons[int(self.oriLabel)], cons[int(self.target)], self.norm, normValue))

    def updateNonTarget(self, advImg, iter):
        cons = getCons(self.model, advImg[0])
        normValue = getNormValue(self.img, advImg, self.norm)
        if (iter <= 0):
            logger.info('origin_con: %.4f, %s: %.3f' % (cons[self.oriLabel], self.norm, normValue))
            print('origin_con: %.4f, %s: %.3f' % (cons[self.oriLabel], self.norm, normValue))
        else:
            logger.info('[Iter: %7d] origin_con: %.4f, %s: %.3f' % (iter, cons[self.oriLabel], self.norm, normValue))
            print('[Iter: %7d] origin_con: %.4f, %s: %.3f' % (iter, cons[self.oriLabel], self.norm, normValue))

    def logEnd(self, advImg):
        if self.isTarget:
            self.endTarget(advImg)
        else:
            self.endNonTarget(advImg)
        saveImage(advImg[0], self.imageName, self.extention, 'adv')
        saveImage(get_diff(self.img[0], advImg[0]), self.imageName, self.extention, 'diff')

    def endTarget(self, advImg):
        cons = getCons(self.model, advImg[0])
        normValue = getNormValue(self.img, advImg, self.norm)
        success = 'successed' if cons.argmax() == self.target else 'failed'
        logger.info('\n*** \n    End of the attack, it %s\n'

                    '    Origin_label: %s\n'
                    '    Target_label: %s\n'
                    '    Current label: %s\n'
                    '    %s: %.3f\n'
                    '***' % (success, self.oriLabelStr,
                             mapLabel(self.databaseName, self.target),
                             mapLabel(self.databaseName, cons.argmax()),
                             self.norm, normValue))
        print('\n*** \n    End of the attack, it %s\n'

              '    Origin_label: %s\n'
              '    Target_label: %s\n'
              '    Current label: %s\n'
              '    %s: %.3f\n'
              '***' % (success, self.oriLabelStr,
                       mapLabel(self.databaseName, self.target),
                       mapLabel(self.databaseName, cons.argmax()),
                       self.norm, normValue))

    def endNonTarget(self, advImg):
        cons = getCons(self.model, advImg[0])
        normValue = getNormValue(self.img, advImg, self.norm)
        success = 'successed' if cons.argmax() != self.oriLabel else 'failed'
        logger.info('\n*** \n    End of the attack, it %s\n'
                    '    Origin_label: %s\n'
                    '    Current label: %s\n'
                    '    %s: %.3f\n'
                    '***' % (success, self.oriLabelStr,
                             mapLabel(self.databaseName, cons.argmax()),
                             self.norm, normValue))
        print('\n*** \n    End of the attack, it %s\n'
              '    Origin_label: %s\n'
              '    Current label: %s\n'
              '    %s: %.3f\n'
              '***' % (success, self.oriLabelStr,
                       mapLabel(self.databaseName, cons.argmax()),
                       self.norm, normValue))
