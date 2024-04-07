import numpy as np
from mmdet.registry import DATASETS
from mmdet.datasets.base_det_dataset import BaseDetDataset
from .scripts.hyundai_reader import HyundaiReader
from typing import List, Union


@DATASETS.register_module()
class HyundaiDataset(BaseDetDataset):
    METAINFO = {
        'classes':
            ['bump', 'manhole', 'steel', 'pothole', 'flat_bump'],
        'palette': [(220, 20, 60), (0, 0, 142), (0, 0, 70), (0, 60, 100), (255, 100, 0)]
    }

    def __init__(self, *args, **kwargs):
        self.data_split = None
        self.train_ratio = 0.8
        self.split_stride = 100
        self.ori_shape = (1200, 1920)
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:

        self.data_split = self.data_prefix['img'].split('/')[-1]
        data_list = []
        hyundai_reader = HyundaiReader(self.data_root, self.data_split)
        indices = np.arange(hyundai_reader.num_frames())

        for idx in indices:
            parsed_data_info = self.parse_hyundai_data_info(idx, hyundai_reader)
            data_list.append(parsed_data_info)

        return data_list

    def parse_hyundai_data_info(self, frame_index, reader) -> Union[dict, List[dict]]:
        data_info = dict()
        data_info['img_path'] = reader.frame_names[frame_index]
        data_info['img_id'] = frame_index
        data_info['seg_map_path'] = None
        data_info['height'] = self.ori_shape[0]
        data_info['width'] = self.ori_shape[1]

        instances = []
        bboxes, categories = reader.get_bboxes(frame_index)
        for bbox_org, category_org in zip(bboxes, categories):
            instance = {}
            if category_org not in self.METAINFO["classes"]:
                continue
            category = self.cat2label(category_org)

            x1, y1, w, h = self.cycxhw2xywh(bbox_org[:4])
            height = bbox_org[4]
            inter_w = max(0, min(x1 + w, data_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, data_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if w * h <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            instance['bbox'] = bbox
            instance['bbox_label'] = category
            instance['height'] = height
            instance['ignore_flag'] = 0
            instances.append(instance)

        data_info['instances'] = instances
        return data_info

    def cycxhw2xywh(self, yxhw: dict):
        cy, cx, h, w = yxhw
        return cx-w/2, cy-h/2, w, h

    def cat2label(self, category):
        return self.METAINFO["classes"].index(category)
