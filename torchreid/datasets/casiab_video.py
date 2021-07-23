from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
# from scipy.misc import imsave

from .bases import BaseVideoDataset


class Casiab(BaseVideoDataset):
    """
    MARS

    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.

    URL: http://www.liangzheng.com.cn/Project/project_mars.html
    
    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 9330 (gallery)
    # cameras: 6
    """
    dataset_dir = 'casiab'

    def __init__(self, root='data', min_seq_len=0, verbose=True, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train_candi')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery_candi')
        # self.track_train_info_path = osp.join(self.dataset_dir, 'info/tracks_train_info.mat')
        # self.track_test_info_path = osp.join(self.dataset_dir, 'info/tracks_test_info.mat')
        self.query_dir = osp.join(self.dataset_dir, 'query_candi')

        self._check_before_run()

        train = self._process_data(self.train_dir, relabel=True, min_seq_len=min_seq_len)
        query = self._process_data(self.query_dir, relabel=False, min_seq_len=min_seq_len)
        gallery = self._process_data(self.gallery_dir, relabel=False, min_seq_len=min_seq_len)

        if verbose:
            print("=> Casiab loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, _, self.num_train_cams = self.get_videodata_info(self.train)
        self.num_query_pids, _, self.num_query_cams = self.get_videodata_info(self.query)
        self.num_gallery_pids, _, self.num_gallery_cams = self.get_videodata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, dir_path, relabel=False, min_seq_len=0):
        pid_paths = glob.glob(osp.join(dir_path, '*'))
        
        tracklets = []
        for pid_path in pid_paths:
            pid = int(osp.basename(pid_path))

            camid_paths = glob.glob(osp.join(pid_path, 'nm*'))
            for camid_path in camid_paths:
                camid = int(osp.basename(camid_path)[4])

                target_view_dir = osp.join(camid_path, '090')
                image_paths = glob.glob(osp.join(target_view_dir, '*'))

                if len(image_paths) >= min_seq_len:
                    img_paths = tuple(image_paths)
                    tracklets.append((image_paths, pid, camid))
        return tracklets

