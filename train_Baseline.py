from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim

from args import argument_parser, image_dataset_kwargs, optimizer_kwargs
from torchreid.data_manager import ImageDataManager
from torchreid import models
from torchreid.losses import CrossEntropyLoss, TripletLoss, DeepSupervision, FullTripletLoss, CFLLoss
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.loggers import Logger, RankLogger
from torchreid.utils.torchtools import count_num_param, open_all_layers, open_specified_layers
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.eval_metrics import evaluate
from torchreid.samplers import RandomIdentitySampler
from torchreid.optimizers import init_optimizer

# global variables
parser = argument_parser()
args = parser.parse_args()


def main():
    global args

    torch.manual_seed(args.seed)
    if not args.use_avai_gpus: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
    log_name = 'log_test.txt' if args.evaluate else 'log_train.txt'
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU, however, GPU is highly recommended")

    print("Initializing image data manager")
    dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    # ReID-Stream:
    print("Initializing ReID-Stream: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dm.num_train_pids, reid_dim=args.reid_dim, loss={'xent', 'htri'})
    print("ReID Model size: {:.3f} M".format(count_num_param(model)))

    criterion_xent = CrossEntropyLoss(num_classes=dm.num_train_pids, use_gpu=use_gpu, label_smooth=args.label_smooth)
    criterion_htri = TripletLoss(margin=args.margin)

    # 2. Optimizer
    # Main ReID-Stream:
    optimizer = init_optimizer(model.parameters(), **optimizer_kwargs(args))
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")

        for name in args.target_names:
            print("Evaluating {} ...".format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']
            distmat = test(model, queryloader, galleryloader, use_gpu, return_distmat=True)

            if args.visualize_ranks:
                visualize_ranked_results(
                    distmat, dm.return_testdataset_by_name(name),
                    save_dir=osp.join(args.save_dir, 'ranked_results', name),
                    topk=20
                )
        return

    start_time = time.time()
    ranklogger = RankLogger(args.source_names, args.target_names)
    train_time = 0
    print("==> Start training")

    for epoch in range(args.start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterion_xent, criterion_htri, \
              optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)

        scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (
                epoch + 1) == args.max_epoch:
            print("==> Test")

            for name in args.target_names:
                print("Evaluating {} ...".format(name))
                queryloader = testloader_dict[name]['query']
                galleryloader = testloader_dict[name]['gallery']
                rank1 = test(model, queryloader, galleryloader, use_gpu)
                ranklogger.write(name, epoch + 1, rank1)

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, False, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    ranklogger.show_summary()


def train(epoch, model, criterion_xent, criterion_htri, \
          optimizer, trainloader, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (imgs, pids, cloth_ids, _, img_paths, masks) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if use_gpu:
            imgs, pids, cloth_ids, masks = imgs.cuda(), pids.cuda(), cloth_ids.cuda(), masks.cuda()

        # gait img 64*64 only have person in the middle, but reid person 64*64 have person in the entire content, so need this pre-processing:
        padding_length = (args.mask_height - args.mask_width) // 2
        left_right_padding = nn.ZeroPad2d((padding_length, padding_length, 0, 0))
        masks = left_right_padding(masks)

        # Main ReID-Stream:
        features, outputs = model(imgs)

        # ReID loss local:
        xent_loss = criterion_xent(outputs, pids)
        htri_loss = criterion_htri(features, pids)

        loss_total = args.loss_ReID_cla_local * xent_loss + args.loss_ReID_tri_local * htri_loss

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        losses.update(loss_total.item(), pids.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses))

        end = time.time()


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_cloth_ids, q_camids = [], [], [], []
        for batch_idx, (imgs, pids, cloth_ids, camids, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs) # 其实第二个应该是gait，但是无所谓，用不到
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_cloth_ids.extend(cloth_ids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_cloth_ids = np.asarray(q_cloth_ids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_cloth_ids, g_camids = [], [], [], []
        for batch_idx, (imgs, pids, cloth_ids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_cloth_ids.extend(cloth_ids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_cloth_ids = np.asarray(g_cloth_ids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch_size))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_cloth_ids, g_cloth_ids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    if return_distmat:
        return distmat
    return cmc[0]


if __name__ == '__main__':
    main()