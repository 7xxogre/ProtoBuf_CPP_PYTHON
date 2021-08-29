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
        img_prefix='./Train/',
        classes=classes,
        ann_file='./Train/annotations.json'),
    val=dict(
        img_prefix='./Validation/',
        classes=classes,
        ann_file='./Validation/annotations.json'),
    test=dict(
        img_prefix='./Test/',
        classes=classes,
        ann_file='./Test/annotations.json'))

work_dir = 'work_dir'

runner = dict(type='EpochBasedRunner', max_epochs=50)
# resume_from = 'work_dir/epoch_16.pth'
checkpoint_config = dict(interval=1, save_optimizer = True)
#checkpoint_config

# python labelme2coco.py Dataset data_dataset_converted --labels labels.txt