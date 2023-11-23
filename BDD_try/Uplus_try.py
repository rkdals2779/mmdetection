import mmcv
import cv2
import numpy

from mmdet.apis import init_detector
from mmengine.runner import Runner
from mmengine.config import Config
from mmdet.utils.rilab.dataloader.readers.mmdet_coco_shape_reader import MMDetCocoShapeReader
from mmdet.utils.rilab.dataloader.readers.dataset_visualizer import DatasetVisualizer


config_file = '../configs/Hyundai/yolox/yolox_s_8xb8-300e_hyundai.py'
# config_file = '../configs/Uplus/yolox/yolox_s_8xb8-300e_uplus.py'
# config_file = '../configs/Uplus/dino/dino-4scale_r50_8xb2-12e_uplus.py'
cfg = Config.fromfile(config_file)
checkpoint_file = '/home/falcon/shin_workspace/Datacleaning/mmdetection/checkpoints/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'


def try_uplus():
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    data_loader = Runner.build_dataloader(model.cfg.train_dataloader)

    for idx, batch_frame_data in enumerate(data_loader):
        classes = data_loader.dataset.metainfo['classes']
        mmreader = MMDetCocoShapeReader(batch_frame_data, classes)
        batch_size = len(batch_frame_data['inputs'])
        vis_cfg = {'image': {},
                   'box2d': {},
                   'lane': {}
                   }
        vs = DatasetVisualizer(mmreader, vis_cfg)
        for i in range(batch_size):
            print(batch_frame_data['data_samples'][i].img_path)
            vs.visualize(i, wait=0)
            bboxed = mmreader.get_box2d(i)
            image = mmreader.get_image(i)
            classes = mmreader.get_class(i)

if __name__ == "__main__":
    try_uplus()



