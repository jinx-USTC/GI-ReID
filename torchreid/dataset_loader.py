from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from PIL import Image
import numpy as np
import os.path as osp
import io

import torch
from torch.utils.data import Dataset
from .transforms import build_transforms_grey, build_transforms_RGB

from args import argument_parser
# global variables
parser = argument_parser()
args = parser.parse_args()

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path)  #.convert('RGB') Gait only have 1 channel
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def read_image_RGB(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None, training=True):
        self.dataset = dataset
        self.transform = transform
        self.training = training
        self.transform_grey = build_transforms_grey(args.mask_height, args.mask_width, is_train=True)
        self.transform_RGB = build_transforms_RGB(args.height, args.width, is_train=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, cloth_id, camid = self.dataset[index]
        img = read_image_RGB(img_path)

        if self.training:
            try: # our extracted masks
                if args.source_names[0] == 'prcc':
                    mask = read_image(img_path.split('train')[0]+'train_mask'+img_path.split('train')[-1].split('.jpg')[0]+'.png')
                elif args.source_names[0] == 'ltcc':
                    mask = read_image(img_path.split('train')[0]+'train_mask'+img_path.split('train')[-1])
            except: # using PRCC provided sketch
                mask = read_image(img_path.split('/rgb/')[0]+'/sketch/'+img_path.split('/rgb/')[-1])

            mask = self.transform_grey(mask)
        
        if self.transform is not None:
            img = self.transform_RGB(img)

        if self.training:
            return img, pid, cloth_id, camid, img_path, mask
        else:
            if args.concat_mask and args.source_names[0] == 'ltcc':
                try:
                    mask = read_image(img_path.split('test')[0] + 'test_mask' + img_path.split('test')[-1])
                except:
                    mask = read_image(img_path.split('query')[0] + 'query_mask' + img_path.split('query')[-1])

                mask = self.transform_grey(mask)
                return img, pid, cloth_id, camid, mask

            return img, pid, cloth_id, camid, img_path


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    _sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample_method='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample_method = sample_method
        self.transform = transform
        self.cut_padding = 10

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample_method == 'random':
            """
            Randomly sample seq_len items from num items,
            if num is smaller than seq_len, then replicate items
            """
            indices = np.arange(num)
            replace = False if num >= self.seq_len else True
            indices = np.random.choice(indices, size=self.seq_len, replace=replace)
            # sort indices to keep temporal order (comment it to be order-agnostic)
            indices = np.sort(indices)
        
        elif self.sample_method == 'evenly':
            """
            Evenly sample seq_len items from num items.
            """
            if num >= self.seq_len:
                num -= num % self.seq_len
                indices = np.arange(0, num, num/self.seq_len)
            else:
                # if num is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num)
                num_pads = self.seq_len - num
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32)*(num-1)])
            assert len(indices) == self.seq_len
        
        elif self.sample_method == 'all':
            """
            Sample all items, seq_len is useless now and batch_size needs
            to be set to 1.
            """
            indices = np.arange(num)
        
        else:
            raise ValueError("Unknown sample method: {}. Expected one of {}".format(self.sample_method, self._sample_methods))

        imgs = []
        img_name = []
        for index in indices:
            img_path = img_paths[int(index)]
            img = read_image(img_path)
            img_name.append(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)

        return imgs, pid, camid, img_name

        # return imgs[:, :, :, self.cut_padding:-self.cut_padding], pid, camid, img_paths
