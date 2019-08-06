
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np


class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths, instance_paths = self.get_paths(opt)
        if self.opt.dataset_mode == 'series':
            self.label_dir = label_paths
            self.image_dir = image_paths
            self.instance_dir = instance_paths
            size = len(self.label_dir)
            self.dataset_size = size
        else:
            util.natural_sort(label_paths)
            util.natural_sort(image_paths)
            if not opt.no_instance:
                util.natural_sort(instance_paths)

            label_paths = label_paths[:opt.max_dataset_size]
            image_paths = image_paths[:opt.max_dataset_size]
            instance_paths = instance_paths[:opt.max_dataset_size]

            if not opt.no_pairing_check:
                for path1, path2 in zip(label_paths, image_paths):
                    assert self.paths_match(path1, path2), \
                        "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

            self.label_paths = label_paths
            self.image_paths = image_paths
            self.instance_paths = instance_paths

            size = len(self.label_paths)
            self.dataset_size = size

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]

        return filename1_without_ext == filename2_without_ext

    def load_image(self, path, is_RGB=False):
        if is_RGB:
            return Image.open(path).convert('RGB')
        else:
            return Image.open(path)

    def __getitem__(self, index):
        # Label Image

        label_path = self.label_paths[index]

        label = self.load_image(label_path)

        if not(_is_pil_image(label) or _is_numpy_image(label)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(
                    type(label)))

        if isinstance(label, np.ndarray):
            params = get_params(self.opt, label.shape)
            # handle numpy array
        else:
            params = get_params(self.opt, label.size)

        if False:
            transform_label = get_transform(
                self.opt,
                params,
                method=Image.NEAREST,
                normalize=self.opt.contain_dontcare_label)
        else:
            transform_label = get_transform(self.opt, params)

        if False:
            label_tensor = transform_label(label) * 255.0
            # 'unknown' is opt.label_nc
            label_tensor[label_tensor == 255] = self.opt.label_nc
        else:
            label_tensor = transform_label(label)

        # input image (real images)
        image_path = self.image_paths[index]
        image = self.load_image(image_path, is_RGB=True)

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = load_image(
                instance_path)                                #
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})
