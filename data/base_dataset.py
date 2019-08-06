

import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random

from batchgenerators.transforms.spatial_transforms import SpatialTransform,\
    MirrorTransforms

from batchgenerators.transforms.color_transforms import ClipValueRange,\
    ContrastAugmentationTransform,\
    BrightnessTransform,\
    BrightnessMultiplicativeTransform,\
    GammaTransform

from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform,\
    RandomCropTransform

from batchgenerators.transforms.sample_normalization_transforms import MeanStdNormalizationTransform,\
    RangeTransform,\
    CutOffOutliersTransform

from batchgenerators.transforms.utility_transforms import NumpyToTensor

from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform

from batchgenerators.transforms.noise_transforms import RicianNoiseTransform,\
    BlankSquareNoiseTransform
from batchgenerators.transforms.abstract_transforms import Compose


# to do : seg processing, data augmentation

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass



def get_transforms(opt, Normalize=False, To_tensor=True):
    transforms = []
    if not opt.no_clip:
        transforms.append(
            ClipValueRange(
                min=opt.clip['min'],
                max=opt.clip['max'],
                data_key="image"))
        transforms.append(
            ClipValueRange(
                min=opt.clip['min'],
                max=opt.clip['max'],
                data_key="label"))

    if not opt.no_minmax:
        transforms.append(RangeTransform(rnge=opt.mm_range, data_key="image"))
        transforms.append(RangeTransform(rnge=opt.mm_range, data_key="label"))

    if Normalize:
        transforms.append(
            MeanStdNormalizationTransform(
                (0.33,), (0.11,), data_key="image"))
        transforms.append(
            MeanStdNormalizationTransform(
                (0.33,), (0.11,), data_key="label"))
        transforms.append(ClipValueRange(min=-1, max=1, data_key="image"))
        transforms.append(ClipValueRange(min=-1, max=1, data_key="label"))

    if 'resize' in opt.preprocess_mode:
        transforms.append(
            ResizeTransform(
                opt.load_size,
                data_key="image",
                label_key="label"))
        
    if 'crop' in opt.preprocess_mode:
        transforms.append(
            RandomCropTransform(
                crop_size=opt.crop_size,
                data_key="image",
                label_key="label"))
        
    SpatialTransform # for rotation
        
    if not opt.cutoff: # for CT
        transforms.append(CutOffOutliersTransform(

    if opt.isTrain and not opt.no_flip:
        transforms.append(MirrorTransform(axis=(1),data_key="image", label_key="label"))
    if To_tensor:
        transforms.append(NumpyToTensor(cast_to='float'))
    return Compose(transforms)


