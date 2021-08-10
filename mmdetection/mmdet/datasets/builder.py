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

# custom collate
def my_collate(batch, samples_per_gpu=1, use_origin=False):
    if use_origin or isinstance(batch[0]['img'], Sequence):
        return collate(batch, samples_per_gpu)
    else:
        ret = dict()
        siz = batch[0]['img'].data.shape
        width = int(siz[1]/2)       # 입력의 width의 half
        height = int(siz[2]/2)      # 입력의 height의 half
        width_lis = [0, width//2, width]
        height_lis = [0, height//2, height]
        sx = [wid_point for _ in range(3) for wid_point in width_lis ]
        sy = [hei_point for hei_point in height_lis for _ in range(3)]

        # 입력 이미지 메타데이터 shape 변경        
        metas_key = 'img_metas'
        for i in range(0, len(batch), samples_per_gpu):
            lst = []
            for j in range(i, i + samples_per_gpu):
                sample = batch[j][metas_key]
                sample.data['img_shape'] = (width, height, sample.data['img_shape'][2])
        
        
        # 데이터를 9분할 하여 저장. 이때 object가 존재하는 부분만 저장하게 됨.
        key = 'img'
        key1 = 'gt_bboxes'
        key2 = 'gt_labels'
        key3 = 'gt_masks'
        metas_stack = []
        stacked = []    # img
        stack1 = []     # gt_bboxes
        stack2 = []     # gt_labels
        stack3 = []     # gt_masks
        for i in range(0, len(batch), samples_per_gpu):
            metas_lst = []
            lst = None  # img Tensor
            lst1 = []   # gt_bboxes list
            lst2 = []   # gt_labels list
            lst3 = []   # gt_masks list, 8 BitmapMasks
            false_cnt = 0
            for j in range(i, i+samples_per_gpu):
                metas_sample = batch[j][metas_key]
                sample = batch[j][key]          # img
                sample1 = batch[j][key1]        # gt_bboxes
                sample2 = batch[j][key2]        # gt_labels
                sample3 = batch[j][key3]        # gt_masks
                for k in range(9):              # divide 9 parts
                    bbox_offset = torch.tensor([sx[k], sy[k], sx[k], sy[k]],
                                        dtype=torch.float32)
                    bboxes = sample1.data - bbox_offset
                    bboxes[:, 0::2] = torch.clamp(bboxes[:, 0::2], 0, width)
                    bboxes[:, 1::2] = torch.clamp(bboxes[:, 1::2], 0, height)
                    valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
                    
                    if valid_inds.any():        # valid object in this part...
                        masking = sample3.data[valid_inds.nonzero(as_tuple = True)[0]].crop(np.asarray([sx[k], sy[k], sx[k] + width, sy[k] + height]))
                        if masking.masks.any().item():     # valid mask in this part...
                            if lst is None:
                                lst = torch.unsqueeze(sample.data[:, sx[k]:sx[k] + width, sy[k]:sy[k]+height], dim = 0)
                            else:
                                lst = torch.cat((lst, torch.unsqueeze(sample.data[:, sx[k]:sx[k] + width, sy[k]:sy[k]+height], dim = 0)), dim = 0)
                            metas_lst.append(metas_sample.data)
                            lst1.append(bboxes[valid_inds, :])
                            lst2.append(sample2.data[valid_inds])
                            lst3.append(masking)
                        else:
                            false_cnt += 1
                    else:
                        false_cnt += 1

            # 부족한 수 만큼 랜덤으로 추가합니다.
            batch_len = len(lst1)
            for j in range(false_cnt):
                idx = np.random.randint(batch_len)
                metas_lst.append(metas_lst[idx])
                lst = torch.cat((lst, torch.unsqueeze(lst[idx, :, :, :], dim = 0)), dim = 0)
                lst1.append(lst1[idx])
                lst2.append(lst2[idx])
                lst3.append(lst3[idx])
            metas_stack.append(metas_lst)
            stacked.append(lst)             # img stack
            stack1.append(lst1)             # gt_bboxes stack
            stack2.append(lst2)             # gt_labels stack
            stack3.append(lst3)             # gt_masks stack
        ret[metas_key] = DataContainer(metas_stack, batch[0][metas_key].stack, batch[0][metas_key].padding_value, cpu_only=True)
        ret[key] = DataContainer(stacked, batch[0][key].stack, batch[0][key].padding_value, cpu_only=False)
        ret[key1] = DataContainer(stack1, batch[0][key1].stack, batch[0][key1].padding_value, cpu_only=False)
        ret[key2] = DataContainer(stack2, batch[0][key2].stack, batch[0][key2].padding_value, cpu_only=False)
        ret[key3] = DataContainer(stack3, batch[0][key3].stack, batch[0][key3].padding_value, cpu_only=True)
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
