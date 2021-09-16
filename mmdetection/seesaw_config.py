_base_ = [
    'configs/_base_/models/cascade_mask_rcnn_r50_fpn.py',
    'configs/_base_/datasets/coco_instance.py',
    'configs/_base_/schedules/schedule_2x.py', 'configs/_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                cls_predictor_cfg=dict(type='NormedLinear', tempearture=20),
                loss_cls=dict(
                    type='SeesawLoss',
                    p=0.8,
                    q=2.0,
                    num_classes=10,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                cls_predictor_cfg=dict(type='NormedLinear', tempearture=20),
                loss_cls=dict(
                    type='SeesawLoss',
                    p=0.8,
                    q=2.0,
                    num_classes=10,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                cls_predictor_cfg=dict(type='NormedLinear', tempearture=20),
                loss_cls=dict(
                    type='SeesawLoss',
                    p=0.8,
                    q=2.0,
                    num_classes=10,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_head=dict(num_classes=10)),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            # LVIS allows up to 300
            max_per_img=300)))

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
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
            type='Albu',
            transforms=[
                        dict(type='CopyPaste', 
                            blend=True, 
                            sigma=1, 
                            pct_objects_paste=0.8, 
                            p=1.
                            )
                        ],
            bbox_params=dict(
                type='BboxParams',
                format='coco',
                min_visibility=0.05)),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(
        img_prefix='./Train/',
        classes=classes,
        ann_file='./Train/annotations.json',
        pipeline = train_pipeline),
    val=dict(
        img_prefix='./Validation/',
        classes=classes,
        ann_file='./Validation/annotations.json',
        pipeline = test_pipeline),
    test=dict(
        img_prefix='./Test/',
        classes=classes,
        ann_file='./Test/annotations.json',
        pipeline = test_pipeline))
evaluation = dict(interval=24, metric=['bbox', 'segm'])