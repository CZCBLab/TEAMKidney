# dataset settings
dataset_type = 'KidneyPanopticDataset'
data_root = '/restricted/projectnb/czproj/aqzou/data/panoptic_ki_1024/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadPanopticAnnotations', backend_args=backend_args),
    # dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PhotoMetricDistortion', brightness_delta=16, contrast_range=(0.8, 1.2), saturation_range=(0.9,1.1), hue_delta=0),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
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
            img='panoptic_train/images_1024/', seg='panoptic_train/segmentation/'),
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
        ann_file='panoptic_val/annotations.json',
        data_prefix=dict(img='panoptic_val/images_1024/', seg='panoptic_val/segmentation/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoPanopticMetric',
    ann_file=data_root + 'panoptic_val/annotations.json',
    seg_prefix=data_root + 'panoptic_val/segmentation/',
    backend_args=backend_args)
test_evaluator = val_evaluator

