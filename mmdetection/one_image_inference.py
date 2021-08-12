import mmcv
import torch
import numpy as np
from PIL import Image


from mmdet.apis import init_detector
from mmdet.apis import inference_detector



def save_predict_obj_img(model, img_path, save_path = None):
    """
        image 경로를 받아 해당 이미지에 대한 모델의 예측을 생성해 object로 추정되는
        crop을 save_path에 저장하는 함수
    """
    if save_path is None:
        save_path = ""
        for p in range(len(img_path.split('\\')) - 1):
            save_path += img_path.split('\\')[p]
            save_path += '\\'

    result = inference_detector(model, img_path)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    img_name = img_path.split('\\')[-1].split('.')[0]
    
    labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
    labels = np.concatenate(labels)

    bboxes = np.vstack(bbox_result)
    img_origin = mmcv.imread(img_path).astype(np.uint8)
    cnt = 1
    for i in range(len(bboxes)):
        if bboxes[i, -1] > 0.4:
            im = Image.fromarray(img_origin[int(bboxes[i, 1]) : int(bboxes[i, 3]),int(bboxes[i, 0]) : int(bboxes[i, 2]),:])
            im.save(save_path + img_name + '_' + str(labels[i]) + '_' + str(cnt) + '.jpg')
            cnt += 1
    


def get_predict(model, img_path):
    """
        이미지 경로를 받아 해당 이미지에 대한 모델의 예측을 생성하여 
        [[class_num, 확률, class_num, bbox 가로, bbox 세로], ...] 의 object별 리스트 리턴
    """
    result = inference_detector(model, img_path)
    
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
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
        if bbox[-1] > 0.4:
            lst = [labels[i],bbox[-1], labels[i], int(bbox[3]) - int(bbox[1]), int(bbox[2]) - int(bbox[0])]
            ret.append(lst)

    return ret

if __name__=="__main__":
    
    checkpoint_path = "work_dir/epoch_50.pth"
    
    save_predict_obj_img("test_data\JPEGImages\C010301_20210726_144327400.jpg")
    get_predict("test_data\JPEGImages\C010301_20210726_144327400.jpg")