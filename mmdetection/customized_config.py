_base_ = './configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=10),
        mask_head=dict(num_classes=10),
    ),
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                        checkpoint='torchvision://resnet101')
    ))

dataset_type = 'COCODataset'
classes = ( 'anabaena',
            'aphanizomenon',
            'aulacoseira',
            'eudorina',
            'fragilaria',
            'microcystis',
            'oscillatoria',
            'pediastrum',
            'staurastrum',
            'synedra')

data = dict(
    train=dict(
        img_prefix='./data_dataset_converted/',
        classes=classes,
        ann_file='./data_dataset_converted/annotations.json',
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ]),
    val=dict(
        img_prefix='./test_data/',
        classes=classes,
        ann_file='./test_data/annotations.json'),
    test=dict(
        img_prefix='./test_data/',
        classes=classes,
        ann_file='./test_data/annotations.json'))

work_dir = 'work_dir'
#load_from = 'work_dir/epoch_24.pth'
runner = dict(type='EpochBasedRunner', max_epochs=50)

#checkpoint_config

