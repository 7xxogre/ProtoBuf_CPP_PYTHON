import mmcv
import torch
import numpy as np
from PIL import Image
import os
import cv2
import json

from mmdet.apis import init_detector
from mmdet.apis import inference_detector



def save_predict_obj_img(model, img_path, save_path = None, percentage = 0.4):
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
    img_origin = mmcv.imread(img_path).astype(np.uint8)
    cnt = 1
    for i in range(len(bboxes)):
        if bboxes[i, -1] > percentage:
            im = Image.fromarray(img_origin[int(bboxes[i, 1]) : int(bboxes[i, 3]),int(bboxes[i, 0]) : int(bboxes[i, 2]),:])
            path = os.path.join(save_path , img_name + '_' + str(labels[i]) + '_' + str(cnt) + '.jpg')
            im.save(path)
            cnt += 1
    

def get_predict(model, img_path, percentage = 0.4):
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
    ret = []
    labels = np.concatenate(labels)
    for i, bbox in enumerate(bboxes):
        if bbox[-1] > percentage:
            lst = [img_path, str(labels[i]),str(bbox[-1])[:8], str(labels[i]), \
                str(int(bbox[3]) - int(bbox[1])), str(int(bbox[2]) - int(bbox[0]))]
            str_lst = ','.join(lst)
            ret.append(str_lst)

    return ret

def get_json(model, img_path, percentage = 0.4):
    result = inference_detector(model, img_path)
    
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0] 
    else:
        bbox_result, segm_result = result, None
    image_name = os.path.split(img_path)[-1].split('.')[0]
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
        with open(image_name + ".json", 'w') as f:
            json.dump(ret_json, f)
    return ret_json



if __name__=="__main__":
    
    checkpoint_path = "work_dir/epoch_50.pth"
    
    save_predict_obj_img("test_data\JPEGImages\C010301_20210726_144327400.jpg")
    get_predict("test_data\JPEGImages\C010301_20210726_144327400.jpg")