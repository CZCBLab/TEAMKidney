# dataset settings
dataset_type = 'KidneyPanopticDataset'
data_root = '/restricted/projectnb/czproj/aqzou/data/panoptic_1024/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadPanopticAnnotations', backend_args=backend_args),
    # dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PhotoMetricDistortion', brightness_delta=16, contrast_range=(0.8, 1.2), saturation_range=(1.0,1.0), hue_delta=0),
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
        ann_file='train/annotations/annotations.json',
        data_prefix=dict(
            img='train/images/', seg='train/panoptic_train/'),
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
        ann_file='val/annotations/annotations.json',
        data_prefix=dict(img='val/images/', seg='val/panoptic_val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoPanopticMetric',
    ann_file=data_root + 'val/annotations/annotations.json',
    seg_prefix=data_root + 'val/panoptic_val/',
    backend_args=backend_args)
test_evaluator = val_evaluator

