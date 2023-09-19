import mmcv
import cv2
import numpy

from mmdet.apis import init_detector
from mmengine.runner import Runner
from mmengine.config import Config
from mmdet.utils.rilab.dataloader.readers.mmdet_coco_shape_reader import MMDetCocoShapeReader
from mmdet.utils.rilab.dataloader.readers.dataset_visualizer import DatasetVisualizer


config_file = '../configs/Uplus/dino/dino-4scale_r50_8xb2-12e_uplus.py'
# config_file = '../configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
cfg = Config.fromfile(config_file)
checkpoint_file = '/home/falcon/shin_workspace/Datacleaning/mmdetection/checkpoints/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'

def try_uplus():
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # visualizer = VISUALIZERS.build(model.cfg.visualizer)
    data_loader = Runner.build_dataloader(model.cfg.train_dataloader)

    for idx, frame_data in enumerate(data_loader):
        image = frame_data['inputs'][0].cpu().numpy().transpose(1, 2, 0)
        data_sample = frame_data['data_samples'][0]
        data_sample2 = frame_data['data_samples'][1]
        gt_instances = data_sample.gt_instances
        bboxes = gt_instances.bboxes
        labels = gt_instances.labels
        cxcywh = bboxes.cxcywh
        numpy_bboxes = bboxes.numpy()
        x1, y1, x2, y2 = numpy.split(numpy_bboxes, 4, axis=-1)
        bboxes_yxhw = numpy.round(numpy.array([(y1 + y2) / 2, (x1 + x2) / 2, y2 - y1, x2 - x1], dtype=numpy.int32)).T
        classes = ('1', '2', '3', '4', '5', '6')
        img_path = frame_data['data_samples'][0].img_path
        image = mmcv.imconvert(image, 'bgr', 'rgb')


        classes = data_loader.dataset.METAINFO['classes']
        mmreader = MMDetCocoShapeReader(frame_data, classes)
        batch_size = len(frame_data['inputs'])
        vis_cfg = {'image': {},
                   'box2d': {},
                   }
        vs = DatasetVisualizer(mmreader, vis_cfg)
        for i in range(batch_size):
            vs.visualize(i, wait=0)
            bboxed = mmreader.get_box2d(i)
            image = mmreader.get_image(i)
            classes = mmreader.get_class(i)

if __name__ == "__main__":
    try_uplus()



