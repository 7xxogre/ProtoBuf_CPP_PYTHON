import copy
import platform
import random
from functools import partial

import numpy as np

from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader

from collections.abc import Sequence
import torch
from mmcv.parallel import collate

from mmcv.parallel import DataContainer
from mmdet.core import BitmapMasks

# for my_single_gpu_test(mmdet\apis\test.py)
def inference_collate(batch, samples_per_gpu):
    batch_copy = copy.deepcopy(batch)
    origin_ret = collate(batch_copy, samples_per_gpu)
    ret = dict()
    metas_key = 'img_metas'
    img_key = 'img'
    ori_siz = batch[0][metas_key][0].data['ori_shape']
    siz = batch[0][metas_key][0].data['img_shape']
    width = int(siz[0]/2)           # input's half width
    height = int(siz[1]/2)          # input's half height
    width_lis = [0, width//2, width]
    height_lis = [0, height//2, height]

    sx = [wid_point for _ in range(3) for wid_point in width_lis ]
    sy = [hei_point for hei_point in height_lis for _ in range(3)]

    metas_stack = []
    img_stack = []      # img

    # modify meta data's 'img_shape'
    for i in range(0, len(batch), samples_per_gpu):
        for j in range(i, i + samples_per_gpu):
            batch[j][metas_key][0].data['img_shape'] = (width, height, siz[2])
            batch[j][metas_key][0].data['pad_shape'] = batch[j][metas_key][0].data['img_shape']
            batch[j][metas_key][0].data['ori_shape'] = (ori_siz[0]//2, ori_siz[1]//2, ori_siz[2])


    for i in range(0, len(batch), samples_per_gpu):
        metas_lst = []
        img_lst = None  # img Tensor
        for j in range(i, i+samples_per_gpu):
            metas_sample = batch[i][metas_key][j]
            sample_img = batch[i][img_key][j]          # img
            for x, y in zip(sx, sy):              # divide 9 parts
                if img_lst is None:
                    img_lst = torch.unsqueeze(sample_img.data[:, x:x + width, y:y + height], dim = 0)
                else:
                    img_lst = torch.cat((img_lst, torch.unsqueeze(sample_img.data[:, x:x + width, y:y + height], dim = 0)), dim = 0)
                metas_lst.append(metas_sample.data)

        metas_stack.append(metas_lst)       
        img_stack.append(img_lst)           # img stack

    ret[metas_key] = [DataContainer(metas_stack,batch[0][metas_key][0].stack, batch[0][metas_key][0].padding_value, cpu_only=True)]
    ret[img_key] = img_stack

    return ret, origin_ret


# custom collate (if use_origin is "True" then this program use mmcv's collate function)
def my_collate(batch, samples_per_gpu=1, use_origin=True):
    rand = np.random.rand()
    if rand >= 0.7:
       use_origin = True
    if use_origin or isinstance(batch[0]['img'], Sequence):
        # return inference_collate(batch, samples_per_gpu)  # <- 접근 방식 5번 (test 방식 변경)
        return collate(batch, samples_per_gpu)
    else:
        ret = dict()
        siz = batch[0]['img'].data.shape
        width = int(siz[1]/2)       # input width's half
        height = int(siz[2]/2)      # input height's half

        width_lis = [0, width//2, width]
        height_lis = [0, height//2, height]
        print(f"siz : {siz}, width : {width}, height : {height}")
        # start points
        sx = [wid_point for _ in range(3) for wid_point in width_lis ]
        sy = [hei_point for hei_point in height_lis for _ in range(3)]
        # sx = [0, 0, width, width]
        # sy = [0, height, 0, height]


        # modify meta data's 'img_shape'
        metas_key = 'img_metas'
        for i in range(0, len(batch)):
            batch[i][metas_key].data['img_shape'] = (width, height, batch[i][metas_key].data['img_shape'][2])
            batch[i][metas_key].data['pad_shape'] = batch[i][metas_key].data['img_shape']
            # batch[i][metas_key].data['ori_shape'] = (batch[i][metas_key].data['ori_shape'][0]/2, batch[i][metas_key].data['ori_shape'][1]/2, batch[i][metas_key].data['ori_shape'][2])
        
        
        # divide image to 9 parts
        img_key = 'img'
        bbox_key = 'gt_bboxes'
        label_key = 'gt_labels'
        mask_key = 'gt_masks'
        
        metas_stack = []
        img_stack = []      # img
        bbox_stack = []     # gt_bboxes
        label_stack = []    # gt_labels
        mask_stack = []     # gt_masks

        for i in range(0, len(batch), samples_per_gpu):
            metas_lst = []
            img_lst = None  # img Tensor
            bbox_lst = []   # gt_bboxes list
            label_lst = []  # gt_labels list
            mask_lst = []   # gt_masks list, 8 BitmapMasks
            false_cnt = 0
            
            for j in range(i, i + samples_per_gpu):
                metas_sample = batch[j][metas_key]
                sample = batch[j][img_key]          # img
                sample1 = batch[j][bbox_key]        # gt_bboxes
                sample2 = batch[j][label_key]       # gt_labels
                sample3 = batch[j][mask_key]        # gt_masks
                origin_cnt = sample3.data.masks.sum(axis = (1,2))           # each object's number of pixels

                for x, y in zip(sx, sy):              # divide 9 parts
                    bbox_offset = torch.tensor([x, y, x, y],
                                        dtype=torch.float32)
                    bboxes = sample1.data - bbox_offset
                    bboxes[:, 0::2] = torch.clamp(bboxes[:, 0::2], 0, width)
                    bboxes[:, 1::2] = torch.clamp(bboxes[:, 1::2], 0, height)
                    valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])

                    if valid_inds.any():        # valid object in this part...
                        masking = sample3.data[valid_inds.nonzero(as_tuple = True)[0]].crop(np.asarray([x, y, x + width, y + height]))
                        mask_cnt = masking.masks.sum(axis = (1, 2))
                        mask_valid_inds = (mask_cnt >= origin_cnt[valid_inds.nonzero(as_tuple = True)[0]] * 0.4)
                        valid_inds[torch.where(valid_inds == True)[0]] = torch.from_numpy(mask_valid_inds)
                        if valid_inds.any():     # valid mask in this part...

                            if img_lst is None:
                                img_lst = torch.unsqueeze(sample.data[:, x:x + width, y:y + height], dim = 0)
                            else:
                                img_lst = torch.cat((img_lst, torch.unsqueeze(sample.data[:, x:x + width, y:y + height], dim = 0)), dim = 0)
                            metas_lst.append(metas_sample.data)
                            bbox_lst.append(bboxes[valid_inds, :])
                            label_lst.append(sample2.data[valid_inds])
                            mask_lst.append(masking[mask_valid_inds])
                        else:
                            false_cnt += 1
                    else:
                        false_cnt += 1

            # random add with transpose
            batch_len = len(bbox_lst)
            false_cnt += 1
            for _ in range(false_cnt//2):
                idx = np.random.randint(batch_len)
                metas_lst.append(metas_lst[idx])
                label_lst.append(label_lst[idx])
                # transpose add
                img_lst = torch.cat((img_lst, torch.unsqueeze(img_lst[idx,:,:,:].transpose(1,2), dim = 0)), dim = 0)
                bbox_temp = torch.zeros_like(bbox_lst[idx])
                bbox_temp[:,0] = bbox_lst[idx][:,1]
                bbox_temp[:, 2] = bbox_lst[idx][:, 3]
                bbox_temp[:, 1] = bbox_lst[idx][:, 0]
                bbox_temp[:, 3] = bbox_lst[idx][:, 2]
                bbox_lst.append(bbox_temp)
                mask_lst.append(BitmapMasks(mask_lst[idx].masks.transpose(0,2,1), mask_lst[idx].width, mask_lst[idx].height))

                # just add
                # img_lst = torch.cat((img_lst, torch.unsqueeze(img_lst[idx, :, :, :], dim = 0)), dim = 0)
                # bbox_lst.append(bbox_lst[idx])
                # mask_lst.append(mask_lst[idx])

            metas_stack.append(metas_lst)
            img_stack.append(img_lst)             # img stack
            bbox_stack.append(bbox_lst)             # gt_bboxes stack
            label_stack.append(label_lst)             # gt_labels stack
            mask_stack.append(mask_lst)             # gt_masks stack

        ret[metas_key] = DataContainer(metas_stack, batch[0][metas_key].stack, batch[0][metas_key].padding_value, cpu_only=True)
        ret[img_key] = DataContainer(img_stack, batch[0][img_key].stack, batch[0][img_key].padding_value, cpu_only=False)
        ret[bbox_key] = DataContainer(bbox_stack, batch[0][bbox_key].stack, batch[0][bbox_key].padding_value, cpu_only=False)
        ret[label_key] = DataContainer(label_stack, batch[0][label_key].stack, batch[0][label_key].padding_value, cpu_only=False)
        ret[mask_key] = DataContainer(mask_stack, batch[0][mask_key].stack, batch[0][mask_key].padding_value, cpu_only=True)
        # print("return:\n",ret)
        return ret

from .samplers import DistributedGroupSampler, DistributedSampler, GroupSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def _concat_dataset(cfg, default_args=None):
    from .dataset_wrappers import ConcatDataset
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    seg_prefixes = cfg.get('seg_prefix', None)
    proposal_files = cfg.get('proposal_file', None)
    separate_eval = cfg.get('separate_eval', True)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        # pop 'separate_eval' since it is not a valid key for common datasets.
        if 'separate_eval' in data_cfg:
            data_cfg.pop('separate_eval')
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets, separate_eval)


def build_dataset(cfg, default_args=None):
    from .dataset_wrappers import (ConcatDataset, RepeatDataset,
                                   ClassBalancedDataset)
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            sampler = DistributedGroupSampler(
                dataset, samples_per_gpu, world_size, rank, seed=seed)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=False, seed=seed)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(my_collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
