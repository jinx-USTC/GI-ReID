# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset

from args import argument_parser
# global variables
parser = argument_parser()
args = parser.parse_args()

class Prcc(BaseImageDataset):
    """
    PRCC
    Reference:
    Person Re-identification by Contour Sketch under Moderate Clothing Change.
    TPAMI-2019

    Dataset statistics:
    # identities: 221, with 3 camera views.
    # images: 150IDs (train) + 71IDs (test)

    Dataset statistics: (A--->C, cross-clothes settings)
      ----------------------------------------
      subset   | # ids | # cloth_ids | # images | # cameras
      ----------------------------------------
      train    |   150 |     3 |    17896 |         3
      query    |    71 |     1 |     3384 |         1
      gallery  |    71 |     1 |     3543 |         1
      ----------------------------------------

    Two test settings:
    parser.add_argument('--cross-clothes', action='store_true',
                        help="the person matching between Camera views A and C was cross-clothes matching")
    parser.add_argument('--same-clothes', action='store_true',
                        help="the person matching between Camera views A and B was performed without clothing changes")
    """
    dataset_dir = 'prcc/rgb' # could change to sketch/contour folder

    def __init__(self, root='/home/jinx/data', verbose=True, **kwargs):
        super(Prcc, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.validation_dir = osp.join(self.dataset_dir, 'val')
        self.probe_gallery_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query, gallery = self._process_test_dir(self.probe_gallery_dir, relabel=False)

        if verbose:
            print("=> PRCC dataset loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_cloth_ids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_cloth_ids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_cloth_ids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.probe_gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.probe_gallery_dir))


    def _process_dir(self, dir_path, relabel=False):

        # Load from train
        pid_dirs_path = glob.glob(osp.join(dir_path, '*'))

        dataset = []
        pid_container = set()
        camid_mapper = {'A': 1, 'B': 2, 'C': 3}
        for pid_dir_path in pid_dirs_path:
            img_paths = glob.glob(osp.join(pid_dir_path, '*.jp*'))
            for img_path in img_paths:
                pid = int(osp.basename(pid_dir_path))
                pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for pid_dir_path in pid_dirs_path:
            img_paths = glob.glob(osp.join(pid_dir_path, '*.jp*'))
            for img_path in img_paths:
                pid = int(osp.basename(pid_dir_path))
                camid = camid_mapper[osp.basename(img_path)[0]]
                cloth_id = camid
                camid -= 1  # index starts from 0
                if relabel: pid = pid2label[pid]
                dataset.append((img_path, pid, cloth_id, camid))

        return dataset

    def _process_test_dir(self, dir_path, relabel=False):

        camid_dirs_path = glob.glob(osp.join(dir_path, '*'))

        query = []
        gallery = []
        pid_container = set()
        camid_mapper = {'A': 1, 'B': 2, 'C': 3}

        for camid_dir_path in camid_dirs_path:
            pid_dir_paths = glob.glob(osp.join(camid_dir_path, '*'))
            for pid_dir_path in pid_dir_paths:
                pid = int(osp.basename(pid_dir_path))
                img_paths = glob.glob(osp.join(pid_dir_path, '*'))
                for img_path in img_paths:
                    camid = camid_mapper[osp.basename(camid_dir_path)]
                    camid -= 1  # index starts from 0
                    if camid == 0:
                        cloth_id = camid
                        query.append((img_path, pid, cloth_id, camid))
                    else:
                        if args.cross_clothes and camid == 2:
                            cloth_id = camid
                            gallery.append((img_path, pid, cloth_id, camid))
                        elif args.same_clothes and camid == 1:
                            cloth_id = camid
                            gallery.append((img_path, pid, cloth_id, camid))

        return query, gallery
