import mmcv
import torch
import numpy as np
from PIL import Image
import os
import cv2
import json

from mmdet.apis import init_detector
from mmdet.apis import inference_detector

# NMS algorithm for all class
def non_max_suppresssion_fast(boxes, overlap_thresh = 0.5):
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

    return pick

def save_predict_obj_img(model, img_path, save_path = None, percentage = 0.4, NMS = False):
    """
        image 경로를 받아 해당 이미지에 대한 모델의 예측을 생성해 object로 추정되는
        crop을 save_path에 저장하는 함수
    """
    if save_path is None:
        save_path = os.path.split(img_path)[0]

    result = inference_detector(model, img_path)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None

    img_name = os.path.split(img_path)[-1].split('.')[0]
    labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(bbox_result)
    

    # 모든 클래스에 nms 알고리즘 적용
    if NMS == True:
        pick = non_max_suppresssion_fast(boxes = bboxes, overlap_thresh = 0.5)
        bboxes = bboxes[pick]
        labels = labels[pick]

    img_origin = mmcv.imread(img_path).astype(np.uint8)
    cnt = 1

    for i in range(len(bboxes)):
        if bboxes[i, -1] > percentage:
            im = Image.fromarray(img_origin[int(bboxes[i, 1]) : int(bboxes[i, 3]),int(bboxes[i, 0]) : int(bboxes[i, 2]),:])
            path = os.path.join(save_path , img_name + '_' + str(labels[i]) + '_' + str(cnt) + '.jpg')
            im.save(path)
            cnt += 1
    

def get_predict(model, img_path, percentage = 0.4, NMS = False):
    """
        이미지 경로를 받아 해당 이미지에 대한 모델의 예측을 생성하여 
        [[class_num, 확률, class_num, bbox 가로, bbox 세로], ...] 의 object별 리스트 리턴
    """
    result = inference_detector(model, img_path)
    
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0] 
    else:
        bbox_result, segm_result = result, None

    bboxes = np.vstack(bbox_result)

    labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
    labels = np.concatenate(labels)

    # 모든 클래스에 nms 알고리즘 적용
    if NMS == True:
        pick = non_max_suppresssion_fast(boxes = bboxes, overlap_thresh = 0.5)
        bboxes = bboxes[pick]
        labels = labels[pick]
    
    ret = []
    
    for i, bbox in enumerate(bboxes):
        if bbox[-1] > percentage:
            lst = [img_path, str(labels[i]),str(bbox[-1])[:8], str(labels[i]), \
                str(int(bbox[3]) - int(bbox[1])), str(int(bbox[2]) - int(bbox[0]))]
            str_lst = ','.join(lst)
            ret.append(str_lst)

    return ret

def get_json(model, img_path, percentage = 0.4, NMS = False):
    result = inference_detector(model, img_path)
    
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0] 
    else:
        bbox_result, segm_result = result, None

    json_name = img_path.split('.jpg')[0]
    bboxes = np.vstack(bbox_result)
    labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
    if segm_result is not None:
        segs = [segm for segm_list in segm_result for segm in segm_list]
    else:
        segs = None

    labels = np.concatenate(labels)
    classes = model.CLASSES
    img_arr = cv2.imread(img_path)

    # NMS 알고리즘 적용
    if NMS == True:
        pick = non_max_suppresssion_fast(boxes = bboxes, overlap_thresh = 0.5)
        bboxes = bboxes[pick]
        labels = labels[pick]
        segs = [segs[idx] for idx in pick]        


    ret_json = dict()
    ret_json["version"] = "4.5.9"
    ret_json["flags"] = dict()
    ret_json["shapes"] = list()
    ret_json["imagePath"] = str(img_path)
    ret_json["imageData"] = None
    ret_json["imageHeight"] = int(img_arr.shape[0])
    ret_json["imageWidth"] = int(img_arr.shape[1])
    ret = []
    for i, bbox in enumerate(bboxes):
        if bbox[-1] > percentage:
            obj = dict()
            obj['label'] = str(classes[labels[i]])
            
            points_lst = []
            if segs is not None:
                
                mask = segs[i]
                conto = np.zeros_like(mask, dtype = np.uint8)
                
                conto[mask == False] = 255
            
                _, imthres = cv2.threshold(conto, 127, 255, cv2.THRESH_BINARY_INV)
                
                contour2, _ = cv2.findContours(imthres, cv2.RETR_EXTERNAL, \
                                                                cv2.CHAIN_APPROX_SIMPLE)
                
                for cord in contour2[0]:
                    points_lst.append([float(cord[0][0]), float(cord[0][1])])
            obj['points'] = points_lst
            obj['group_id'] = None
            obj['shape_type'] = "polygon"
            obj['flags'] = dict()
            ret_json['shapes'].append(obj)

    with open(json_name + ".json", 'w') as f:
        json.dump(ret_json, f)

    return ret_json




if __name__=="__main__":
    
    checkpoint_path = "work_dir/epoch_50.pth"
    
    # save_predict_obj_img("test_data\JPEGImages\C010301_20210726_144327400.jpg")
    # get_predict("test_data\JPEGImages\C010301_20210726_144327400.jpg")

    from mmcv import Config
    cfg = Config.fromfile("customized_config.py")
    model = init_detector(cfg, checkpoint=checkpoint_path)

    get_json(model,"../Original_converted/converted/JPEGImages/155_data.jpg", NMS = True)