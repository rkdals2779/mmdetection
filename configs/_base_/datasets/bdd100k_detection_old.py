_base_ = './coco_detection.py'

METAINFO = {
    'classes':
    ('person', 'car', 'rider', 'bus', 'truck', 'bike', 'motor',
     'traffic light', 'traffic sign', 'train', ),

    # palette is a list of color tuples, which is used for visualization.
    'palette': [(220, 20, 60), (0, 0, 142), (197, 226, 255), (0, 60, 100),
                (0, 0, 70), (119, 11, 32), (0, 0, 230),
                (255, 179, 240), (0, 125, 92), (0, 80, 100), ]
}

data_root = '/media/falcon/IanBook8T/datasets/bdd100k/'

train_dataloader = dict(
    dataset=dict(data_root=data_root,
                 data_prefix=dict(img='bdd100k_images_100k/bdd100k/images/100k/train/'),
                 metainfo=METAINFO,
                 ann_file=data_root + 'annotation/bdd100k_labels_images_det_coco_train.json')
)
val_dataloader = dict(
    dataset=dict(data_root=data_root,
                 data_prefix=dict(img='bdd100k_images_100k/bdd100k/images/100k/val/'),
                 metainfo=METAINFO,
                 ann_file=data_root + 'annotation/bdd100k_labels_images_det_coco_val.json')
)
test_dataloader = dict(
    dataset=dict(data_root=data_root,
                 data_prefix=dict(img='bdd100k_images_100k/bdd100k/images/100k/test/'),
                 metainfo=METAINFO,
                 ann_file=data_root + 'annotation/bdd100k_labels_images_det_coco_val.json')
)
val_evaluator = dict(
    ann_file=data_root + 'annotation/bdd100k_labels_images_det_coco_val.json',
)
test_evaluator = dict(
    ann_file=data_root + 'annotation/bdd100k_labels_images_det_coco_val.json',
)