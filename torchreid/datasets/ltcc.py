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


class Ltcc(BaseImageDataset):
    """
    LTCC_ReID dataset
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 152 (+1 for background)
    # images:  (train) +  (query) +  (gallery)
    """
    dataset_dir = 'LTCC_ReID'

    def __init__(self, root='/home/jinx/data', verbose=True, **kwargs):
        super(Ltcc, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()

        self.cloth_unchange_id_train_list = []
        self.cloth_change_id_train_list = []
        self.cloth_unchange_id_test_list = []
        self.cloth_change_id_test_list = []
        file = open(osp.join(self.dataset_dir, 'info/cloth-unchange_id_train.txt'))
        cloth_unchange_id_train_list = file.readlines()
        for i in cloth_unchange_id_train_list:
            self.cloth_unchange_id_train_list.append(int(i.strip('\n')))
        file = open(osp.join(self.dataset_dir, 'info/cloth-change_id_train.txt'))
        cloth_change_id_train_list = file.readlines()
        for i in cloth_change_id_train_list:
            self.cloth_change_id_train_list.append(int(i.strip('\n')))
        file = open(osp.join(self.dataset_dir, 'info/cloth-unchange_id_test.txt'))
        cloth_unchange_id_test_list = file.readlines()
        for i in cloth_unchange_id_test_list:
            self.cloth_unchange_id_test_list.append(int(i.strip('\n')))
        file = open(osp.join(self.dataset_dir, 'info/cloth-change_id_test.txt'))
        cloth_change_id_test_list = file.readlines()
        for i in cloth_change_id_test_list:
            self.cloth_change_id_test_list.append(int(i.strip('\n')))

        train = self._process_dir_train(self.train_dir, self.cloth_unchange_id_train_list, self.cloth_change_id_train_list, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> LTCC_ReID loaded")
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
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir_train(self, dir_path, cloth_unchange_id_train_list, cloth_change_id_train_list, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'([\d]+)_([\d]+)+_c(\d)')  # cloth ID 也要，所以返回三个！！re.compile(r'([\d]+)_[\d]+_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            img_name = osp.basename(img_path)
            pid, _, _ = map(int, pattern.search(img_name).groups())
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            img_name = osp.basename(img_path)
            pid, cloth_id, camid = map(int, pattern.search(img_name).groups())
            assert 0 <= pid <= 151  # pid == 0 means background
            assert 1 <= cloth_id <= 14
            assert 1 <= camid <= 12
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            if args.train_with_all_cloth:
                dataset.append((img_path, pid, cloth_id, camid))
            elif args.train_with_only_cloth_changing:
                if pid in cloth_change_id_train_list:
                    dataset.append((img_path, pid, cloth_id, camid))

        return dataset

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'([\d]+)_([\d]+)+_c(\d)')  # cloth ID 也要，所以返回三个！！re.compile(r'([\d]+)_[\d]+_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            img_name = osp.basename(img_path)
            pid, _, _ = map(int, pattern.search(img_name).groups())
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            img_name = osp.basename(img_path)
            pid, cloth_id, camid = map(int, pattern.search(img_name).groups())
            assert 0 <= pid <= 151  # pid == 0 means background
            assert 1 <= cloth_id <= 14
            assert 1 <= camid <= 12
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, cloth_id, camid))

        return dataset