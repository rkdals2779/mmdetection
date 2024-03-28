from mmdet.apis import init_detector, inference_detector
import config as cfg
from dataloader import Dataloader
from visualization import Visualization

import time


class Test:
    def __init__(self):
        self.dataloader = Dataloader(cfg.DATASET_PATH, cfg.BATCH_SIZE)
        self.mmdet_config_path = cfg.MMDET_CONFIG_PATH
        self.mmdet_checkpoint_path = cfg.MMDET_CHECKPOINT_PATH
        self.model = init_detector(self.mmdet_config_path, self.mmdet_checkpoint_path, device='cuda:0')
        self.visualization = Visualization()
        self.batch_img_list = None

    def main_test(self):
        self.data_load()
        pred_start = time.time()
        for num, batch_img_path in enumerate(self.batch_img_list):
            inference_results = self.image_inference(batch_img_path)
            for img_path, result in zip(batch_img_path, inference_results):
                pred = result.pred_instances
                self.visualization(img_path, pred)
        pred_end = time.time()
        print(f"prediction time: {pred_end - pred_start}")

    def image_inference(self, image):
        return inference_detector(self.model, image)

    pred_end = time.time()

    def data_load(self):
        self.batch_img_list = self.dataloader.imgs_list_load()


if __name__ == '__main__':
    main_start = time.time()
    test = Test()
    test.main_test()
    main_end = time.time()
    print('test time: ', main_end - main_start)

