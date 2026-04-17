# dataset settings
dataset_type = 'KidneyPanopticDataset'
data_root = '/restricted/projectnb/czproj/aqzou/data/panoptic_ki_2624/'

backend_args = None
image_size = (2624, 2624)
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadPanopticAnnotations', backend_args=backend_args),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.5, 1.5),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=crop_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='PhotoMetricDistortion', brightness_delta=16, contrast_range=(0.8, 1.2), saturation_range=(0.9,1.1), hue_delta=0),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    # dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadPanopticAnnotations', backend_args=backend_args),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True), #  DefaultSampler
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='panoptic_train/annotations.json',
        data_prefix=dict(
            img='panoptic_train/train_images_2624/', seg='panoptic_train/segmentation/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='panoptic_val_1024/annotations.json',
        # data_prefix=dict(img='panoptic_val_1024/images_1024_jpg/', seg='panoptic_val_1024/segmentation/'),
        ann_file='panoptic_val_2624/annotations.json',
        data_prefix=dict(img='panoptic_val_2624/val_images_2624/', seg='panoptic_val_2624/segmentation/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoPanopticMetric',
    ann_file=data_root + 'panoptic_val_2624/annotations.json',
    seg_prefix=data_root + 'panoptic_val_2624/segmentation/',
    backend_args=backend_args)
test_evaluator = val_evaluator

