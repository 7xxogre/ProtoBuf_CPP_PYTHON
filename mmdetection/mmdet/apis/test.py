import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results

import matplotlib.pyplot as plt
import numpy as np

# NMS algorithm
def non_max_suppresssion_fast(boxes, masks, overlap_thresh = 0.25):
    if not boxes.size:
        return np.empty((0,), dtype=int)
    if boxes.dtype.kind == 'i':
        boxes = boxes.astype('float')
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(boxes[:, 4])
    while idxs.size:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    return boxes[pick], masks[pick]

# custom test funcion
def my_single_gpu_test(model,
                      data_loader,
                      show=False,
                      out_dir=None,
                      show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    ori_siz = dataset[0]['img_metas'][0].data['ori_shape']
    w_half = ori_siz[0]//2
    h_half = ori_siz[1]//2
    width_lis = [0, w_half//2, w_half]
    height_lis = [0, h_half//2, h_half]
    sx = [wid_point for _ in range(3) for wid_point in width_lis ]
    sy = [hei_point for hei_point in height_lis for _ in range(3)]
    # half_idx = set([1,3,4,5,7])
    # sx = [0, w_half]
    # sy = [0, h_half]

    
    for i, data in enumerate(data_loader):
        partition, origin_data = data
        # get results
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **partition)
            origin_result = model(return_loss=False, rescale=True, **origin_data)

        batch_size = len(result)//9
        classnum = len(result[0][0])
        # 일단 편의를 위해 batch size = 1이라 가정
        bboxes = [np.zeros((0,5)) for _ in range(classnum)]
        masks = [[] for _ in range(classnum)]
        
        # merge results into one img
        for j, cor in enumerate(zip(sx, sy)):
            x, y = cor
            result[j] = list(result[j])
            result[j][0] = list(map(lambda t: t + np.array([y, x, y, x, 0.0]) if len(t) != 0 else t, result[j][0]))

            for k, arr in enumerate(result[j][0]):                    
                if len(bboxes[k]) == 0 and len(arr) != 0:
                    bboxes[k] = arr
                    
                elif len(bboxes[k]) != 0 and len(arr) != 0:
                    bboxes[k] = np.vstack((bboxes[k], arr))
            
            for k, mask_lst in enumerate(result[j][1]):
                for mask in mask_lst:
                    ori_mask = np.zeros((ori_siz[0], ori_siz[1]), dtype = bool)
                    ori_mask[x:x + w_half, y:y + h_half] = mask
                    if len(masks[k]) == 0:
                        masks[k] = np.expand_dims(ori_mask, axis = 0)
                    else:
                        masks[k] = np.concatenate((masks[k], np.expand_dims(ori_mask, axis = 0)), axis = 0)

        # merge with origin result
        origin_bboxes, origin_masks = origin_result[0]
        for i, (bbox, masklst) in enumerate(zip(origin_bboxes, origin_masks)):
            if len(masklst):
                bboxes[i] = np.vstack((bboxes[i], bbox))
                for mask in masklst:
                    if len(masks[i]) == 0:
                        masks[i] = np.expand_dims(mask, axis = 0)
                    else:
                        masks[i] = np.concatenate((masks[i], np.expand_dims(mask, axis = 0)), axis = 0)
        # apply NMS algorithm (class 단위로)
        for i in range(len(bboxes)):
            if len(masks[i]) != 0:
                bboxes[i], masks[i] = non_max_suppresssion_fast(bboxes[i], masks[i])
        # changa mask format (batch of numpy array -> list(numpy array))
        mask_list = []
        for mask in masks:
            temp_list = []
            for mask_partition in mask:
                temp_list.append(mask_partition)
            mask_list.append(temp_list)
        merged_result = [(bboxes, mask_list)]
        
        # show results
        if show or out_dir:
            if batch_size == 1 and isinstance(origin_data['img'][0], torch.Tensor):
                img_tensor = origin_data['img'][0]
            else:
                img_tensor = origin_data['img'][0].data[0]
            img_metas = origin_data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']                
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    merged_result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)
        # encode mask results
        if isinstance(merged_result[0], tuple):
            merged_result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in merged_result]
        results.extend(merged_result)
        
        for _ in range(batch_size):
            prog_bar.update()
    return results

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    # if use inferece_collate...
    # return my_single_gpu_test(model, data_loader, show, out_dir, show_score_thr)

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
