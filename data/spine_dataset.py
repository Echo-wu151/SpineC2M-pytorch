
from data.base_dataset import get_transforms
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
from pydicom import dcmread
import os
import torch
import numpy as np


class SpineDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='crop')
        parser.set_defaults(label_nc=1)
        parser.set_defaults(no_label=True)
        parser.set_defaults(no_pairing_check=True)
        parser.set_defaults(no_instance=True)
        parser.set_defaults(tf_log=True)
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        return parser

    def load_image(self, path):
        return dcmread(path).pixel_array

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]

        return int(filename1_without_ext) == int(filename2_without_ext)

    def get_paths(self, opt):
        root_phase = os.path.join(opt.dataroot, opt.phase)
        root_dirs = [f.path for f in os.scandir(root_phase) if f.is_dir()]
        label_paths = []
        image_paths = []
        instance_paths = []
        for _dir in root_dirs:
            label_dir = os.path.join(_dir, 'CTA nce')
            image_dir = os.path.join(_dir, 'CTA ce')
            if len(opt.instance_dir) > 0:
                instance_dir = opt.instance_dir
                instance_paths += make_dataset(instance_dir,
                                               recursive=False, read_cache=True)
            label_paths += make_dataset(label_dir,
                                        recursive=False, read_cache=True)
            image_paths += make_dataset(image_dir,
                                        recursive=False, read_cache=True)

        assert len(label_paths) == len(
            image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        return label_paths, image_paths, instance_paths

    def expand_dims(self, dict, keys: tuple):
        for k, d in dict.items():
            if k in keys:
                dict[k] = d[None, None, :]
        return dict

    def __getitem__(self, index):
        # Label Image

        label_path = self.label_paths[index]
        label = self.load_image(label_path)
        #params = get_params(self.opt, label.shape)

        # input image (real images)
        image_path = self.image_paths[index]
        image = self.load_image(image_path)
        #print(    "image {}, {} , label {}, {} ".format(image.min(),image.max(),label.min(),label.max() ) )

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

        input_dict = {'label': label,
                      'instance': instance_tensor,
                      'image': image,
                      'path': image_path,
                      }
        input_dict = self.expand_dims(input_dict, keys=("image", "label"))
        # Give subclasses a chance to modify the final output
        transform = get_transforms_4_dcm(self.opt)

        input_dict = transform(**input_dict)

        self.postprocess(input_dict)

        return input_dict
