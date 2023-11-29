import sys
import numpy as np
sys.path.append("./hyundai_script/")
import mmdet.datasets.hyundai_script.hyundai_config as cfg
from mmdet.datasets.hyundai_script.hyundai_reader_manager import HyundaiReader, HyundaiDriveManager
from typing import List, Union

from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class HyundaiDataset(BaseDetDataset):
    METAINFO = {
        'classes':
            cfg.Datasets.Hyundai.CATEGORIES_TO_USE,  # Define your classes
        'palette': [(220, 20, 60), (0, 0, 142), (0, 0, 70), (0, 60, 100),
                    (0, 0, 230), (250, 170, 30), (119, 11, 32), (179, 0, 194),
                    (209, 99, 106), (5, 121, 0), (227, 255, 205),
                    (200, 200, 200), (150, 150, 150), ]  # Define your palette
    }

    def load_data_list(self) -> List[dict]:
        dataset_cfg = cfg.Datasets.Hyundai

        data_list = []

        mode = self.data_prefix['img'].split('/')[-1]

        img_id = 1
        drive_mngr = HyundaiDriveManager(self.data_root, mode)
        drive_paths = drive_mngr.get_drive_paths()
        for drive_path in drive_paths:
            hyundai_reader = HyundaiReader(drive_path, mode, dataset_cfg)
            for idx in range(hyundai_reader.num_frames()):
                info_dict = {"img_id": img_id, "frame_index": idx, "reader": hyundai_reader}
                parsed_data_info = self.parse_data_info(info_dict)
                img_id += 1
                data_list.append(parsed_data_info)

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        img_id = raw_data_info['img_id']
        idx = raw_data_info['frame_index']
        hyundai_reader = raw_data_info["reader"]

        data_info = {}

        data_info['img_path'] = hyundai_reader.frame_names[idx]
        data_info['img_id'] = img_id
        data_info['seg_map_path'] = None
        data_info['height'] = cfg.Datasets.Hyundai.ORI_SHAPE[0]
        data_info['width'] = cfg.Datasets.Hyundai.ORI_SHAPE[1]

        instances = []
        bboxes, categories = hyundai_reader.get_bboxes(idx)
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
