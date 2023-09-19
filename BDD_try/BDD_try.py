import mmcv
from pprint import PrettyPrinter
from mmdet.apis import init_detector
from mmdet.registry import VISUALIZERS
from mmengine.runner import Runner
from mmengine.config import Config
import torch

config_file = '../configs/BDD100K/dino/dino-4scale_r50_8xb2-12e_bdd100k.py'
cfg = Config.fromfile(config_file)
checkpoint_file = '/home/falcon/shin_workspace/Datacleaning/mmdetection/checkpoints/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'
def try_bdd100k():
    pprinter = PrettyPrinter()
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # visualizer = VISUALIZERS.build(model.cfg.visualizer)
    data_loader = Runner.build_dataloader(model.cfg.train_dataloader)
    pprinter.pprint(cfg.train_dataloader)

    for idx, frame_data in enumerate(data_loader):
        print(f'========== {idx:03d} ==========')
        image = frame_data['inputs'][0].cpu().numpy().transpose(1, 2, 0)
        data_sample = frame_data['data_samples'][0]
        print('image shape:', image.shape)
        print('image path:', data_sample.img_path)


        # intead of 'inference_detector(model, image)'
        result = model.test_step(frame_data)
        result = result[0]

        if idx == 0:
            print('---------- frame data\n', frame_data)
            print('---------- prediction\n', result)
        print('bbox shape:', result.gt_instances.bboxes.shape, result.pred_instances.bboxes.shape)

        image = mmcv.imconvert(image, 'bgr', 'rgb')
        # opencv resize: (width, height)
        ori_shape = (data_sample.ori_shape[1], data_sample.ori_shape[0])
        image = mmcv.imresize(image, ori_shape)
        print('resized to original image shape:', image.shape)
        visualizer.dataset_meta = model.dataset_meta
        visualizer.add_datasample(
            'balloon',
            image,
            data_sample=result,
            draw_gt=True,
            show=True)
    print()

try_bdd100k()
